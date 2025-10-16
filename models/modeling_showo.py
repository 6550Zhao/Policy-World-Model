# coding=utf-8
# Copyright 2024 NUS Show Lab, HuggingFace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, mask_by_random_topk
from .phi import PhiForCausalLM
from tqdm import tqdm
from models.compressive_vq_model import Compressive_magvit_v2, MAGVITv2

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "Compressive_magvit_v2":
        return Compressive_magvit_v2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

def get_first_target_values(data, max_batch_index, N):
    batch_indices, target_values = data
    seen_batches = set()
    first_target_values = []


    for batch, target in zip(batch_indices, target_values):
        if batch.item() not in seen_batches:
            seen_batches.add(batch.item())
            first_target_values.append((batch.item(), target.item()))


    for batch in range(max_batch_index):
        if batch not in seen_batches:
            first_target_values.append((batch, N))


    first_target_values.sort(key=lambda x: x[0])


    return [x[1] for x in first_target_values]
class Showo(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            w_clip_vit,
            vocab_size,
            llm_vocab_size,
            llm_model_path='',
            codebook_size=8192,
            num_vq_tokens=256,
            load_from_showo=True,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        if load_from_showo:
            config = AutoConfig.from_pretrained(llm_model_path)#llm
            self.showo = PhiForCausalLM(config)
        else:
            self.showo = PhiForCausalLM.from_pretrained(llm_model_path, attn_implementation='sdpa')
        self.showo.resize_token_embeddings(self.vocab_size)
        self.output_size = self.vocab_size

        if self.w_clip_vit:
            self.mm_projector = torch.nn.Sequential(
                torch.nn.Linear(1024, 2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 2048)
            )

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True
    def init_action(self, sequence_length, out_dim):
        # Action query token
        hidden_size = self.showo.config.hidden_size
        self.action_queries = torch.nn.Embedding(sequence_length, hidden_size)
        self.action_queries.weight.data.fill_(0)
        # Action prediction
        self.pred_act_mlps = torch.nn.ModuleList([
            torch.nn.Linear(hidden_size, hidden_size//2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size//2, hidden_size//2)])
        self.pred_trajectory = torch.nn.Linear(hidden_size // 2, out_dim)
        self.tj_criterion = torch.nn.L1Loss()  # use L1 Loss

    def init_ego(self, sequence_length=13):
        hidden_size = self.showo.config.hidden_size
        self.ego_mlps = torch.nn.ModuleList([
            torch.nn.Linear(sequence_length, hidden_size//2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size//2, hidden_size)])
    def init_cmd(self, sequence_length=3):
        hidden_size = self.showo.config.hidden_size
        self.cmd_queries = torch.nn.Embedding(sequence_length, hidden_size)


    def action_forward(self, x):
        for layer in self.pred_act_mlps:
            x = layer(x)

        return self.pred_trajectory(x)
    def ego_forward(self, x):
        for layer in self.ego_mlps:
            x = layer(x)  #used as ego input

        return x
    def resize_dynamic_size(self, dynamic_size, stage='sft',config=None):

        existing_embedding = self.showo.get_input_embeddings()
        existing_weights = existing_embedding.weight.data

        # current coodbook size
        current_vocab_size, embedding_dim = existing_weights.size()


        if dynamic_size > current_vocab_size:
            raise ValueError("additional_size exceeds the current embedding size.")
        # init new embeddings
        new_embedding_weights = existing_weights[-(dynamic_size+1):-1].clone()


        expanded_weights = torch.cat([existing_weights, new_embedding_weights], dim=0)
        # update codebook

        new_vocab_size = current_vocab_size + dynamic_size
        new_embedding_layer = torch.nn.Embedding(new_vocab_size, embedding_dim)
        new_embedding_layer.weight.data = expanded_weights

        # replace embedding of model
        self.showo.set_input_embeddings(new_embedding_layer)

        # update vocab_size of model
        self.vocab_size += dynamic_size
        self.output_size += dynamic_size

        #lm_head resize

        head_weight = self.showo.lm_head.weight.data #(v,c)
        hidden_size_head = head_weight.size(1)
        head_bias = self.showo.lm_head.bias.data
        init_add_head_weights = head_weight[-(dynamic_size + 1):-1].clone()
        init_add_head_bias = head_bias[-(dynamic_size + 1):-1].clone()
        expanded_head_weights = torch.cat([head_weight, init_add_head_weights], dim=0)
        expanded_head_bias = torch.cat([head_bias, init_add_head_bias], dim=0)
        self.showo.lm_head = torch.nn.Linear(hidden_size_head, self.output_size, bias=True)
        self.showo.lm_head.weight.data = expanded_head_weights
        self.showo.lm_head.bias.data = expanded_head_bias
        if config.dataset.dataset_use == 'sft_nuscenes':
            self.init_action(sequence_length=6, out_dim=2)
            if config.experiment.add_ego:
                print(f"adding ego status for {config.dataset.dataset_use}")
                self.init_ego(sequence_length=13)
            if config.experiment.add_cmd:
                print(f"adding high level command for {config.dataset.dataset_use}")
                self.init_cmd()
        elif config.dataset.dataset_use == 'sft_navsim':
            self.init_action(sequence_length=8, out_dim=3)
            if config.experiment.add_ego:
                print(f"adding ego status and cmd for {config.dataset.dataset_use}")
                self.init_ego(sequence_length=8)
        else:
            pass

    def prepare_inputs_for_generation(self):
        pass

    def forward(
            self,
            input_ids,
            input_embeddings=None,
            attention_mask=None,
            labels=None,
            spcial_dict=None,
            label_smoothing=0.0,
            batch_size_t2d=1,
            batch_size_d2t=1,
            batch_size_lm=1,
            batch_size_mmu=1,
            max_seq_length=128,
            labels_mask_text=None,
            labels_mask_image=None,
            Img_idx=None,
            action_idx=None,
            text_idx=None,
            tokenizer_len=None,
            past_key_values=None,

            **kwargs,
    ):

        if labels is None:
            infer_output = self.showo(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
            logits = infer_output["logits"]
            past_keys_values = infer_output["past_key_values"]

            return logits, past_keys_values
        else:
            if input_embeddings is None:
                logits = self.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
            else:
                logits = self.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']#(8,L,C)


            input_d2t, input_lm, input_mmu = torch.split(input_ids, [batch_size_d2t, batch_size_lm, batch_size_mmu], dim=0)
            logits_d2t, logits_lm, logits_mmu = torch.split(logits, [batch_size_d2t, batch_size_lm, batch_size_mmu], dim=0)
            labels_d2t, labels_lm, labels_mmu = torch.split(labels, [batch_size_d2t, batch_size_lm, batch_size_mmu], dim=0)

            #d2t
            sod_d2t = torch.where(input_d2t[0]==spcial_dict['<|sod|>'].to(input_d2t.device))[-1]
            eod_d2t = torch.where(input_d2t[0] == spcial_dict['<|eod|>'].to(input_d2t.device))[-1]
            soi_d2t = torch.where(input_d2t[0] == spcial_dict['<|soi|>'].to(input_d2t.device))[-1]
            eoi_d2t = torch.where(input_d2t[0] == spcial_dict['<|eoi|>'].to(input_d2t.device))[-1]

            loss_d2t_prev = F.cross_entropy(
                logits_d2t[:,eoi_d2t[0]+1:soi_d2t[1]].contiguous().view(-1, self.output_size), #
                labels_d2t[:,eoi_d2t[0]+1:soi_d2t[1]].contiguous().view(-1), ignore_index=-100, reduce=False,#
            )
            amplify_weight = labels_d2t[:,eoi_d2t[0]+1:soi_d2t[1]].contiguous() == input_d2t[:,eoi_d2t[0]+1:soi_d2t[1]].contiguous()
            amplify_weight = amplify_weight.contiguous().view(-1)
            weights_dynamic = amplify_weight * 0.4 + (~amplify_weight) * 1
            loss_d2t_prev = sum(loss_d2t_prev * weights_dynamic.to(loss_d2t_prev.dtype)) / len(loss_d2t_prev)



            loss_d2t_next = F.cross_entropy(
                logits_d2t[:,eoi_d2t[1]+1:eod_d2t[-1]+1].contiguous().view(-1, self.output_size), #
                labels_d2t[:,eoi_d2t[1]+1:eod_d2t[-1]+1].contiguous().view(-1), ignore_index=-100,reduce=False, #
            )
            amplify_weight = labels_d2t[:,eoi_d2t[1]+1:eod_d2t[-1]+1].contiguous() == input_d2t[:,eoi_d2t[1]+1:eod_d2t[-1]+1].contiguous()
            amplify_weight = amplify_weight.contiguous().view(-1)
            weights_dynamic = amplify_weight * 0.4 + (~amplify_weight) * 1
            loss_d2t_next = sum(loss_d2t_next * weights_dynamic.to(loss_d2t_next.dtype)) / len(loss_d2t_next)


            loss_d2t_text = F.cross_entropy(
                    logits_d2t[:, eod_d2t[-1]+1:-1].contiguous().view(-1, self.output_size),
                    labels_d2t[:, eod_d2t[-1]+1+1:].contiguous().view(-1), ignore_index=-100,
            )
            loss_d2t = loss_d2t_prev+loss_d2t_next+loss_d2t_text

            #lm
            loss_lm = F.cross_entropy(
                    logits_lm[:, :-1].contiguous().view(-1, self.output_size),
                    labels_lm[:, 1:].contiguous().view(-1), ignore_index=-100,
            )
            #mmu
            loss_mmu = F.cross_entropy(
                    logits_mmu[:, :-1].contiguous().view(-1, self.output_size),
                    labels_mmu[:, 1:].contiguous().view(-1), ignore_index=-100,
            )
            return logits, loss_d2t, loss_lm, loss_mmu, input_d2t, logits_d2t

        return logits
    def nus_forward(
        self,
        inputs,
        attention_mask = None,
        labels = None,
        batch_size = None,
        action_len = None,
        sptids_dict= None,
        past_key_values = None,
        gt_tj = None,
        gen_type = None,
        motion_weight=False,
        ego_status=None,
        H_cmd=None,
        mode='nusc',
        plan_mask=None,
        nfp_coffe = None,
        ** kwargs,
    ):
        if labels is None:#infer
            if gen_type not in ['trj','embed']:
                infer_output = self.showo(input_ids=inputs, attention_mask=attention_mask, past_key_values=past_key_values,
                                          use_cache=True, output_hidden_states=True)
            else:
                infer_output = self.showo(inputs_embeds=inputs, attention_mask=attention_mask, past_key_values=past_key_values,
                                          use_cache=True, output_hidden_states=True)
            logits = infer_output["logits"]
            past_keys_values = infer_output["past_key_values"]
            hidden_states = infer_output['hidden_states'][-1]
            return logits, past_keys_values, hidden_states



        else:#train

            input_embeddings = self.showo.model.embed_tokens(inputs)
            # replace action queries
            act_queries = self.action_queries(torch.arange(action_len)[None, ...].repeat(batch_size, 1).to(inputs.device))
            act_queries = act_queries.to(input_embeddings.dtype)
            gt_tj = gt_tj.to(input_embeddings.dtype)
            #add status or cmd token embeddings
            pad_info = 0
            mmu_index = torch.where(inputs == sptids_dict['<|sot|>'].to(inputs.device))[1].unique()
            eod_img_d = torch.where(inputs == sptids_dict['<|eod|>'].to(inputs.device))[1].unique()

            if ego_status is not None:
                ego_token = self.ego_forward(ego_status.to(input_embeddings.dtype))
                input_embeddings[:, mmu_index[0]-1, :] = ego_token
                pad_info = 1
            if H_cmd is not None:
                cmd_queries = self.cmd_queries(H_cmd).to(input_embeddings.dtype)
                input_embeddings[:, mmu_index[0]-1-pad_info, :] = cmd_queries
            input_embeddings[:, -action_len:, :] = act_queries





            assert len(mmu_index) == 2, f"Expected 2 unique values in sot_index, but got {mmu_index}"
            if input_embeddings is None:
                outputs = self.showo(input_ids=inputs, attention_mask=attention_mask, output_hidden_states=True)
            else:
                outputs = self.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)  # (8,L,C)

            logits, hidden_states = outputs['logits'], outputs['hidden_states'][-1]
            # 1. qa loss
            loss_qa = F.cross_entropy(
                logits[:, mmu_index[0]:mmu_index[1]-1].contiguous().view(-1, self.output_size),
                labels[:, mmu_index[0]+1:mmu_index[1]].contiguous().view(-1), ignore_index=-100,
            )
            # 2. video loss
            if mode == 'nusc':
                if motion_weight == True:
                    loss_dynamic = F.cross_entropy(
                        logits[:, mmu_index[-1]+1:eod_img_d[-1]+1].contiguous().view(-1, self.output_size),  #
                        labels[:, mmu_index[-1]+1:eod_img_d[-1]+1].contiguous().view(-1), ignore_index=-100,  #
                        reduce=False,
                    )

                    amplify_weight = labels[:, mmu_index[-1]+1:eod_img_d[-1]+1].contiguous() == inputs[:, mmu_index[-1]+1:eod_img_d[-1]+1].contiguous()
                    amplify_weight = amplify_weight.view(-1)
                    weights_dynamic = amplify_weight * nfp_coffe.beta_coffe + (~amplify_weight) * nfp_coffe.alpha_coffe #amplify_weight * 0.4 + (~amplify_weight) * 1
                    loss_dynamic = sum(loss_dynamic * weights_dynamic.to(loss_dynamic.dtype)) / len(loss_dynamic)
                else:
                    loss_dynamic = F.cross_entropy(
                        logits[:, mmu_index[-1]+1:eod_img_d[-1]+1].contiguous().view(-1, self.output_size),  #
                        labels[:, mmu_index[-1]+1:eod_img_d[-1]+1].contiguous().view(-1), ignore_index=-100,  #
                    )
                # 3. act loss
                logits_tj = self.action_forward(hidden_states[:, -6:])
                if plan_mask is not None:
                    plan_mask=plan_mask.to(device=logits_tj.device, dtype=logits_tj.dtype)
                    loss_tj = self.tj_criterion(logits_tj*plan_mask, gt_tj)
                else:
                    loss_tj = self.tj_criterion(logits_tj, gt_tj)
                return logits, loss_qa, loss_dynamic, loss_tj, mmu_index, eod_img_d
            else:
                # 3. act loss
                logits_tj = self.action_forward(hidden_states[:, -6:])
                loss_tj = self.tj_criterion(logits_tj, gt_tj)

                return logits, loss_qa, loss_tj, mmu_index, eod_img_d

        return logits
    def navsim_forward(
        self,
        inputs,
        attention_mask = None,
        labels = None,
        batch_size = None,
        action_len = None,
        sptids_dict= None,
        past_key_values = None,
        gt_tj = None,
        gen_type = None,
        motion_weight=False,
        ego_status=None,
        mode='navsim',
        frame_bias = 4.0,
        nfp_coffe=None,
        ** kwargs,
    ):
        if labels is None:#infer
            if gen_type not in ['trj','embed']:
                infer_output = self.showo(input_ids=inputs, attention_mask=attention_mask, past_key_values=past_key_values,
                                          use_cache=True, output_hidden_states=True)
            else:
                infer_output = self.showo(inputs_embeds=inputs, attention_mask=attention_mask, past_key_values=past_key_values,
                                          use_cache=True, output_hidden_states=True)
            logits = infer_output["logits"]
            past_keys_values = infer_output["past_key_values"]
            hidden_states = infer_output['hidden_states'][-1]
            return logits, past_keys_values, hidden_states
        else:#train

            input_embeddings = self.showo.model.embed_tokens(inputs)
            # replace action queries
            act_queries = self.action_queries(torch.arange(action_len)[None, ...].repeat(batch_size, 1).to(inputs.device))
            act_queries = act_queries.to(input_embeddings.dtype)
            gt_tj = gt_tj.to(input_embeddings.dtype)
            #add status or cmd token embeddings
            mmu_index = torch.where(inputs == sptids_dict['<|sot|>'].to(inputs.device))[1].unique()
            eod_img_d = torch.where(inputs == sptids_dict['<|eod|>'].to(inputs.device))[1].unique()

            ego_token = self.ego_forward(ego_status.to(input_embeddings.dtype))
            input_embeddings[:, mmu_index[0]-1, :] = ego_token
            input_embeddings[:, -action_len:, :] = act_queries

            assert len(mmu_index) == 2, f"Expected 2 unique values in sot_index, but got {mmu_index}"
            if input_embeddings is None:
                outputs = self.showo(input_ids=inputs, attention_mask=attention_mask, output_hidden_states=True)
            else:
                outputs = self.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask, output_hidden_states=True)  # (8,L,C)

            logits, hidden_states = outputs['logits'], outputs['hidden_states'][-1]
            # 1. qa loss
            # loss_qa = F.cross_entropy(
            #     logits[:, mmu_index[0]:mmu_index[1]-1].contiguous().view(-1, self.output_size),
            #     labels[:, mmu_index[0]+1:mmu_index[1]].contiguous().view(-1), ignore_index=-100,
            # )
            # 2. video loss
            if mode == 'navsim':
                if motion_weight == True:
                    loss_dynamic = F.cross_entropy(
                        logits[:, mmu_index[-1]+1:eod_img_d[-1]+1].contiguous().view(-1, self.output_size),  #
                        labels[:, mmu_index[-1]+1:eod_img_d[-1]+1].contiguous().view(-1), ignore_index=-100,  #
                        reduce=False,
                    )

                    amplify_weight = labels[:, mmu_index[-1]+1:eod_img_d[-1]+1].contiguous() == inputs[:, mmu_index[-1]+1:eod_img_d[-1]+1].contiguous()
                    amplify_weight = amplify_weight.view(-1)
                    weights_dynamic = amplify_weight * nfp_coffe.beta_coffe + (~amplify_weight) * nfp_coffe.alpha_coffe #weights_dynamic = amplify_weight * 0.4 + (~amplify_weight) * 1
                    loss_dynamic = sum(loss_dynamic * weights_dynamic.to(loss_dynamic.dtype)) / len(loss_dynamic)
                else:
                    loss_dynamic = F.cross_entropy(
                        logits[:, mmu_index[-1]+1:eod_img_d[-1]+1].contiguous().view(-1, self.output_size),  #
                        labels[:, mmu_index[-1]+1:eod_img_d[-1]+1].contiguous().view(-1), ignore_index=-100,  #
                    )
                # 3. act loss
                logits_tj = self.action_forward(hidden_states[:, -8:])
                loss_tj = self.tj_criterion(logits_tj, gt_tj)
                return logits, loss_dynamic, loss_tj, mmu_index, eod_img_d
            else:
                # 3. act loss
                logits_tj = self.action_forward(hidden_states[:, -8:])
                loss_tj = self.tj_criterion(logits_tj, gt_tj)

                return logits, loss_tj, mmu_index, eod_img_d

        return logits
    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        num_vq_tokens = config.model.showo.num_vq_tokens
        num_new_special_tokens = config.model.showo.num_new_special_tokens

        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()#得到图像的初始token ids,qu
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,#找到图像的初始msak token ids
                                                    mask_token_id,#torch.where(condition, x, y):torch.where 是 PyTorch 的一个函数，类似于条件运算符。如果 condition 为 True，返回 x，否则返回 y
                                                    input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]#获取无条件的所有text token ids

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)#返回预测结果
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]#从llm_vocab_size+num_new_special_tokens后面的embedding中找索引
            else:
                logits = self(input_ids, attention_mask=attention_mask)
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]

            probs = logits.softmax(dim=-1)#image的codebook是8192
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])


            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))#
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])#
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(#
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)#
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + config.model.showo.llm_vocab_size
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, attention_mask=None, max_new_tokens=100, temperature=1.0, top_k=None, eot_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        try:
            device = idx.device
        except:
            device = input_embeddings.device

        result = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # logits, _ = self(idx_cond)
            logits = self(idx, input_embeddings=input_embeddings, attention_mask=attention_mask)

            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack(#加一列
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(#加一行
                [
                    attention_mask_a,  # L, L+1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
            # append sampled index to the running sequence and continue
            if self.config.w_clip_vit:
                idx_next_embeddings = self.showo.model.embed_tokens(idx_next)
                input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            else:
                idx = torch.cat((idx, idx_next), dim=1)

            if eot_token is not None and idx_next.cpu() == eot_token:
                break

        return result
    # def nusc_gen(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     uncond_input_ids: torch.LongTensor = None,
    #     attention_mask=None,
    #     init_next_frame_tokens=None,
    #     config=None,
    #     gen_frames=11,
    #     sod_ids=None,
    #     labels=None,
    #     action_len=6,
    #     uni_prompting=None,
    #     gt_tj=None,  # only training (x,y)
    #     max_token=3000,
    #     max_text_token=30, #200,
    #     mode='nusc_wo_d',
    #     ego_status=None,
    #     H_cmd=None,
    #     ** kwargs,
    # ):
    #     # begin with all image token ids masked
    #     B = input_ids.shape[0]
    #     condition_len = len(torch.where(input_ids == uni_prompting.sptids_dict['<|eod|>'].to(input_ids.device))[1].unique())
    #     assert condition_len < 12, print(condition_len) # context_length
    #     cur_input = input_ids
    #     attention_mask_step_input = attention_mask
    #     if uncond_input_ids is not None:
    #         uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]  # 获取无条件的所有text token ids
    #     gentext_token = []
    #     genframe_token = []
    #     pred_trj = []
    #     text_trj_length = 1
    #     total_length = cur_input.shape[-1]
    #     input_len = cur_input.shape[-1]
    #     past_key_values = None
    #     gen_type = 'desc'
    #     mask_dtype = attention_mask.dtype
    #     mask_device = attention_mask.device
    #     frame_num = init_next_frame_tokens.shape[-1] if init_next_frame_tokens is not None else None
    #     attention_mask_one_frame = torch.zeros((B, 1, frame_num, frame_num),dtype=torch.bool,device=mask_device) if init_next_frame_tokens is not None else None
    #     attention_mask_one_token = torch.zeros((B, 1, text_trj_length, text_trj_length), dtype=torch.bool, device=mask_device)
    #     stat_num = 1 if ego_status is not None else 0
    #     cmd_num = 1 if H_cmd is not None else 0
    #     attention_mask_action_token = self.showo.model.embed_tokens(uni_prompting.sptids_dict['<|act|>'].to(input_ids.device))[None,...].repeat(B, 1, 1)
    #     text_end_set = set()
    #     act_queries = None
    #     in_if_ = 0
    #     while(total_length < max_token):
    #             #start text generation -> video generation ->traj generation
    #         if (total_length >= (input_len+max_text_token) and in_if_ == 0) or len(text_end_set) == B:# and len(text_end_set) >= 0
    #             text_end_set = set()
    #             in_if_ += 1
    #             if in_if_ >= 2:
    #                 print("error................ ", total_length, text_end_set, in_if_, input_len)
    #             if gen_frames>0:
    #                 gen_type = 'frame'
    #             else:
    #                 attention_mask_step_input = attention_mask_step_input[...,:-1]
    #                 eot_index = None
    #                 continue
    #             cur_input = init_next_frame_tokens
    #             eot_index = get_first_target_values(torch.where(torch.stack(gentext_token, dim=-1) == uni_prompting.sptids_dict['<|eot|>'].to(input_ids.device)), B, len(gentext_token))#当生成的长度超过最大长度的时候，没生成完的样本最后没有eot token
    #             total_length = total_length-1
    #             attention_mask_prev = torch.arange(total_length)[None, None, None, ...].repeat(B, 1, frame_num, 1).to(mask_device)>(input_ids.shape[-1] + torch.tensor(eot_index)[..., None, None, None]).to(mask_device)
    #
    #             attention_mask_step_input = torch.cat((attention_mask_prev, attention_mask_one_frame), -1)#True 为mask
    #             attention_mask_step_input = attention_mask_step_input.to(mask_dtype)  # 1表示mask，0表示不mask
    #             attention_mask_step_input = attention_mask_step_input.masked_fill(
    #                 attention_mask_step_input.to(torch.bool), torch.iinfo(torch.int64).min
    #             ).to(mask_dtype)
    #         elif (len(genframe_token) >= gen_frames) and act_queries is None and in_if_ == 1:
    #             gen_type = 'trj'
    #             act_queries_list = [attention_mask_action_token]
    #             if H_cmd is not None:
    #                 act_queries_list.append(self.cmd_queries(H_cmd).unsqueeze(1))
    #             if ego_status is not None:
    #                 act_queries_list.append(self.ego_forward(ego_status).unsqueeze(1))
    #
    #             act_queries_list.append(self.action_queries(torch.arange(action_len)[None, ...].repeat(B, 1).to(cur_input.device)))
    #             act_queries = torch.cat(act_queries_list, dim=1)
    #             action_chunk = 0
    #             total_length += 1
    #             cur_input = act_queries[:, action_chunk].unsqueeze(1)
    #             attention_mask_step_input = torch.cat((attention_mask_step_input[:, :, -1].unsqueeze(1), attention_mask_one_token), -1)
    #
    #         # else:
    #         #     print("error ", total_length, gentext_token, text_end_set)
    #         logits, past_key_values, hidden_outputs = self.nus_forward(inputs=cur_input,
    #                                                                    attention_mask=attention_mask_step_input,
    #                                                                    past_key_values=past_key_values,
    #                                                                    gen_type=gen_type)
    #         if gen_type == 'desc':
    #             pred_logits = logits[:, -1, :config.model.showo.vocab_size]
    #             total_length += 1
    #             probs = pred_logits.softmax(dim=-1).argmax(-1)
    #             cur_input = probs[..., None]
    #             # uni_prompting.text_tokenizer.batch_decode(probs, skip_special_tokens=True) #visual the str output
    #             attention_mask_step_input = torch.cat((attention_mask_step_input[:,:, -1].unsqueeze(2), attention_mask_one_token), -1)
    #             gentext_token.append(probs) #finally torch.stack(gentext_token,dim=-1)
    #             is_end = torch.where(uni_prompting.sptids_dict['<|eot|>'].to(input_ids.device) == gentext_token[-1])[-1]
    #             # print("gentext_token",gentext_token)
    #             if is_end.numel() != 0:
    #                 text_end_set.update(is_end.tolist())
    #
    #         elif gen_type == 'frame':
    #             pred_logits = logits[:, :, config.model.showo.vocab_size:]
    #             total_length += pred_logits.shape[1]
    #             probs = pred_logits[:, 1:-1].softmax(dim=-1).argmax(-1)  # dynamic的codebook是8192
    #
    #             cur_input = torch.cat(
    #                 [uni_prompting.sptids_dict['<|sod|>'][None, ...].to(cur_input.device).repeat(B, 1),
    #                  probs + config.model.showo.vocab_size,
    #                  uni_prompting.sptids_dict['<|eod|>'][None, ...].to(cur_input.device).repeat(B, 1)], dim=-1)
    #             # attention_mask_step_input = attention_mask_step_input.to(mask_dtype)  # 1表示mask，0表示不mask
    #             # attention_mask_step_input = attention_mask_step_input.masked_fill(attention_mask_step_input.to(torch.bool), torch.iinfo(torch.int64).min).to(mask_dtype)
    #             genframe_token.append(cur_input)
    #             if len(genframe_token) < gen_frames:
    #                 attention_mask_step_input = torch.cat((attention_mask_step_input, attention_mask_one_frame),-1)  # True 为mask
    #         elif gen_type == 'trj':
    #             if action_chunk > 0+stat_num+cmd_num:
    #                 pred_trj.append(self.action_forward(hidden_outputs[:, -1]))
    #                 if action_chunk >= action_len+stat_num+cmd_num:
    #                     break
    #             action_chunk += 1
    #             cur_input = act_queries[:, action_chunk].unsqueeze(1)
    #             attention_mask_step_input = torch.cat((attention_mask_step_input[:, :, -1].unsqueeze(2), attention_mask_one_token), -1)
    #     # assert len(gentext_token) > 0, print("gentext_token is empty at the end of generation", total_length, text_end_set)
    #     # print("in_if_", in_if_),print("text_end_set", text_end_set)
    #     # assert len(genframe_token) > 0, print("genframe_token is empty at the end of generation", gentext_token, pred_trj)
    #     # assert len(pred_trj) ==6, "len of pred_trj is error at the end of generation"
    #     return (torch.stack(gentext_token, -1) if gentext_token is not None else None,
    #             torch.stack(genframe_token, -2).flatten(1, 2) if len(genframe_token)>0 is not None else None,
    #             torch.stack(pred_trj, -2) if pred_trj is not None else None,
    #             eot_index)
    def nusc_gen(
        self,
        input_embed: torch.LongTensor = None,
        uncond_input_ids: torch.LongTensor = None,
        attention_mask=None,
        init_next_frame_embed=None,
        config=None,
        gen_frames=11, #11
        sod_ids=None,
        labels=None,
        action_len=6,
        uni_prompting=None,
        gt_tj=None,  # only training (x,y)
        max_token=3000,
        max_text_token=30,#30, #200,
        ** kwargs,
    ):
        B = input_embed.shape[0]

        cur_input = input_embed
        attention_mask_step_input = attention_mask
        gentext_token = []
        genframe_token = []
        pred_trj = []
        text_trj_length = 1
        total_length = cur_input.shape[1]
        input_len = cur_input.shape[1]
        past_key_values = None
        gen_type = 'desc'
        mask_dtype = attention_mask.dtype
        mask_device = attention_mask.device
        frame_num = init_next_frame_embed.shape[-2] if init_next_frame_embed is not None else None
        attention_mask_one_frame = torch.zeros((B, 1, frame_num, frame_num),dtype=torch.bool,device=mask_device) if init_next_frame_embed is not None else None
        attention_mask_one_token = torch.zeros((B, 1, text_trj_length, text_trj_length), dtype=torch.bool, device=mask_device)
        # stat_num = 1 if ego_status is not None else 0
        # cmd_num = 1 if H_cmd is not None else 0
        attention_mask_action_token = self.showo.model.embed_tokens(uni_prompting.sptids_dict['<|act|>'].to(input_embed.device))[None,...].repeat(B, 1, 1)
        text_end_set = set()
        act_queries = None
        in_if_ = 0
        while(total_length < max_token):
                #start text generation -> video generation ->traj generation
            if (total_length >= (input_len+max_text_token) and in_if_ == 0) or len(text_end_set) == B:# and len(text_end_set) >= 0
                text_end_set = set()
                in_if_ += 1
                if in_if_ >= 2:
                    print("enter more than one time ", total_length, text_end_set, in_if_, input_len)
                if gen_frames>0:
                    gen_type = 'frame'
                else:
                    attention_mask_step_input = attention_mask_step_input[..., :-1]
                    eot_index = None
                    continue
                cur_input = init_next_frame_embed
                eot_index = get_first_target_values(torch.where(torch.stack(gentext_token, dim=-1) == uni_prompting.sptids_dict['<|eot|>'].to(input_embed.device)), B, len(gentext_token))#当生成的长度超过最大长度的时候，没生成完的样本最后没有eot token
                total_length = total_length-1
                attention_mask_prev = torch.arange(total_length)[None, None, None, ...].repeat(B, 1, frame_num, 1).to(mask_device)>(input_embed.shape[-1] + torch.tensor(eot_index)[..., None, None, None]).to(mask_device)

                attention_mask_step_input = torch.cat((attention_mask_prev, attention_mask_one_frame), -1)#True 为mask
                attention_mask_step_input = attention_mask_step_input.to(mask_dtype)  # 1表示mask，0表示不mask
                attention_mask_step_input = attention_mask_step_input.masked_fill(
                    attention_mask_step_input.to(torch.bool), torch.iinfo(torch.int64).min
                ).to(mask_dtype)
            elif (len(genframe_token) >= gen_frames) and act_queries is None and in_if_ == 1:
                gen_type = 'trj'
                act_queries_list = [attention_mask_action_token]
                act_queries_list.append(self.action_queries(torch.arange(action_len)[None, ...].repeat(B, 1).to(cur_input.device)))
                act_queries = torch.cat(act_queries_list, dim=1)
                action_chunk = 0
                total_length += 1
                cur_input = act_queries[:, action_chunk].unsqueeze(1)
                attention_mask_step_input = torch.cat((attention_mask_step_input[:, :, -1].unsqueeze(1), attention_mask_one_token), -1)

            logits, past_key_values, hidden_outputs = self.nus_forward(inputs=cur_input,
                                                                       attention_mask=attention_mask_step_input,
                                                                       past_key_values=past_key_values,
                                                                       gen_type='embed')
            if gen_type == 'desc':
                pred_logits = logits[:, -1, :config.model.showo.vocab_size]
                total_length += 1
                probs = pred_logits.softmax(dim=-1).argmax(-1)
                cur_input = self.showo.model.embed_tokens(probs[..., None])#probs[..., None]
                # uni_prompting.text_tokenizer.batch_decode(probs, skip_special_tokens=True) #visual the str output
                attention_mask_step_input = torch.cat((attention_mask_step_input[:,:, -1].unsqueeze(2), attention_mask_one_token), -1)
                gentext_token.append(probs) #finally torch.stack(gentext_token,dim=-1)
                is_end = torch.where(uni_prompting.sptids_dict['<|eot|>'].to(input_embed.device) == gentext_token[-1])[-1]
                # print("gentext_token",gentext_token)
                if is_end.numel() != 0:
                    text_end_set.update(is_end.tolist())

            elif gen_type == 'frame':
                pred_logits = logits[:, :, config.model.showo.vocab_size:]
                total_length += pred_logits.shape[1]
                probs = pred_logits[:, 1:-1].softmax(dim=-1).argmax(-1)  # dynamic的codebook是8192
                cur_input = torch.cat(
                    [uni_prompting.sptids_dict['<|sod|>'][None, ...].to(cur_input.device).repeat(B, 1),
                     probs + config.model.showo.vocab_size,
                     uni_prompting.sptids_dict['<|eod|>'][None, ...].to(cur_input.device).repeat(B, 1)], dim=-1)
                # attention_mask_step_input = attention_mask_step_input.to(mask_dtype)  # 1表示mask，0表示不mask
                # attention_mask_step_input = attention_mask_step_input.masked_fill(attention_mask_step_input.to(torch.bool), torch.iinfo(torch.int64).min).to(mask_dtype)
                genframe_token.append(cur_input)
                cur_input = self.showo.model.embed_tokens(cur_input)
                if len(genframe_token) < gen_frames:
                    attention_mask_step_input = torch.cat((attention_mask_step_input, attention_mask_one_frame),-1)  # True 为mask
            elif gen_type == 'trj':
                if action_chunk > 0:
                    pred_trj.append(self.action_forward(hidden_outputs[:, -1]))
                    if action_chunk >= action_len:
                        break
                action_chunk += 1
                cur_input = act_queries[:, action_chunk].unsqueeze(1)
                attention_mask_step_input = torch.cat((attention_mask_step_input[:, :, -1].unsqueeze(2), attention_mask_one_token), -1)
        return (torch.stack(gentext_token, -1) if gentext_token is not None else None,
                torch.stack(genframe_token, -2).flatten(1, 2) if len(genframe_token) > 0 is not None else None,
                torch.stack(pred_trj, -2) if pred_trj is not None else None,
                eot_index,
                )
    def navsim_gen(
        self,
        input_embed: torch.LongTensor = None,
        uncond_input_ids: torch.LongTensor = None,
        attention_mask=None,
        config=None,
        gen_frames=10, #10,
        action_len=8,
        uni_prompting=None,
        max_token=3000,
        max_text_token=0,#30, #200,
        ** kwargs,
    ):
        # begin with all image token ids masked
        B = input_embed.shape[0]

        cur_input = input_embed
        attention_mask_step_input = attention_mask
        # if uncond_input_ids is not None:
        #     uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]  # 获取无条件的所有text token ids
        gentext_token = []
        genframe_token = []
        pred_trj = []
        text_trj_length = 1
        total_length = cur_input.shape[1]
        input_len = cur_input.shape[1]
        past_key_values = None
        gen_type = 'frame'
        mask_device = attention_mask.device
        frame_num = 30
        attention_mask_one_frame = torch.zeros((B, 1, frame_num, frame_num),dtype=torch.bool,device=mask_device)
        attention_mask_one_token = torch.zeros((B, 1, text_trj_length, text_trj_length), dtype=torch.bool, device=mask_device)
        attention_mask_action_token = self.showo.model.embed_tokens(uni_prompting.sptids_dict['<|act|>'].to(input_embed.device))[None,...].repeat(B, 1, 1)
        act_queries = None

        while(total_length < max_token):
                #start text generation -> video generation ->traj generation
            if (len(genframe_token) >= gen_frames) and act_queries == None:
                gen_type = 'trj'
                act_queries_list = [attention_mask_action_token]
                act_queries_list.append(self.action_queries(torch.arange(action_len)[None, ...].repeat(B, 1).to(cur_input.device)))
                act_queries = torch.cat(act_queries_list, dim=1)
                action_chunk = 0
                total_length += 1
                cur_input = act_queries[:, action_chunk].unsqueeze(1)
                attention_mask_step_input = torch.cat((attention_mask_step_input[:, :, -1].unsqueeze(1), attention_mask_one_token), -1)

            logits, past_key_values, hidden_outputs = self.navsim_forward(inputs=cur_input,
                                                                           attention_mask=attention_mask_step_input,
                                                                           past_key_values=past_key_values,
                                                                           gen_type='embed')

            if gen_type == 'frame':
                pred_logits = logits[:, -frame_num:, config.model.showo.vocab_size:]
                total_length += pred_logits.shape[1]
                probs = pred_logits[:, 1:-1].softmax(dim=-1).argmax(-1)  # dynamic的codebook是8192

                cur_input = torch.cat(
                    [uni_prompting.sptids_dict['<|sod|>'][None, ...].to(cur_input.device).repeat(B, 1),
                     probs + config.model.showo.vocab_size,
                     uni_prompting.sptids_dict['<|eod|>'][None, ...].to(cur_input.device).repeat(B, 1)], dim=-1)
                genframe_token.append(cur_input)
                cur_input = self.showo.model.embed_tokens(cur_input)
                if len(genframe_token) < gen_frames:
                    attention_mask_step_input = torch.cat((attention_mask_step_input[:, :, -frame_num:], attention_mask_one_frame),-1)  # True 为mask
            elif gen_type == 'trj':
                if action_chunk > 0:
                    pred_trj.append(self.action_forward(hidden_outputs[:, -1]))
                    if action_chunk >= action_len:
                        break
                action_chunk += 1
                cur_input = act_queries[:, action_chunk].unsqueeze(1)
                attention_mask_step_input = torch.cat((attention_mask_step_input[:, :, -1].unsqueeze(2), attention_mask_one_token), -1)

        return torch.stack(genframe_token, -2).flatten(1, 2) if len(genframe_token) > 0 is not None else None, torch.stack(pred_trj, -2) if pred_trj is not None else None