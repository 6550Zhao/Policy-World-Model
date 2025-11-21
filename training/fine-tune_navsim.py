# coding=utf-8
# Copyright 2024 HuggingFace, NUS Show Lab.
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
import moviepy
from moviepy.editor import ImageSequenceClip
import imageio
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union
import copy
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
import random
from models.video_metric import Evaluator
from torch.optim import AdamW
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from models.video_metric import FeatureStats
from data_utils.pwm_dataset import DatasetNavsim
from models import Showo
from models.modeling_showo import get_vq_model_class
from models.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_t2d, create_attention_mask_for_nusc
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from training.utils import flatten_omega_conf, AverageMeter
SYSTEM_PROMPT_LEN = 28
from navsim.pdsm_test_utils import PDSM_eval, pdsm_score_process


try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls) and 'mm_projector' not in name.split('.'):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1]) #unique linear layer name

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    if 'embed_tokens' in lora_module_names:
        lora_module_names.remove('embed_tokens')
    return list(lora_module_names)
def batch_forward(batch_size, input, forward, context_length=None, special_token=None, verbose=False):
    if context_length is None and special_token is None:
        return torch.cat([forward(input[i: i + batch_size], ) for i in trange(0, input.shape[0], batch_size, disable=not verbose)], dim=0)
    else:
        return torch.cat(
            [forward(input[i: i + batch_size], context_length=context_length, special_token=special_token ) for i in trange(0, input.shape[0], batch_size, disable=not verbose)],
            dim=0)

def img_token2pixel(image_tokens_ori, uni_prompting, vq_model, gen_image_token_ids=None):
    if gen_image_token_ids is not None:
        img_token = dict(context=image_tokens_ori["context"], dynamic=gen_image_token_ids)
    else:
        img_token = image_tokens_ori
    img_pixel, _ = vq_model.detokenize(img_token,
                                      offset_tokenzier=len(uni_prompting.text_tokenizer),
                                      sptids_dict=uni_prompting.sptids_dict,
                                      )  # (T-1,C,H,W)
    img_pixel = torch.clamp((img_pixel + 1.0) / 2.0, min=0.0, max=1.0)
    return img_pixel
def video_concate(o_images, r_images, p_images, context_length=None):
    len_o = len(o_images)
    len_r = len(r_images)
    len_p = len(p_images)

    max_len = max(len_o, len_r, len_p)
    t2d_v = []
    for i in range(max_len):
        i_o = o_images[i % len_o]
        i_r = r_images[i % len_r]
        i_p = p_images[i % len_p]
        t2d_v.append(np.concatenate((i_o, i_r, i_p), axis=-2))
    return t2d_v
def main():
    # torch.cuda.empty_cache()
    #########################
    # SETUP Accelerator     #
    #########################
    config_path = 'configs/sft_navsim/navsim.yaml'
    config = OmegaConf.load(config_path)
    config.worker = OmegaConf.load(config.worker)
    # config = get_config()
    # Enable TF32 on Ampere GPU
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    log_dir = os.path.join(config.experiment.output_dir, "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=log_dir,
        split_batches=True,
    )
    if accelerator.is_local_main_process:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir,f"{time_str}_train.log") if accelerator.is_local_main_process else None
    if accelerator.is_local_main_process:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            "%m/%d/%Y %H:%M:%S"
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logging.getLogger().addHandler(stream_handler)

    os.environ["WANDB_MODE"] = "offline"  # debug
    total_batch_size_per_gpu = config.training.batch_size_train_nus #must have context frame tokens
    total_batch_size = config.training.batch_size_train_nus* config.training.gradient_accumulation_steps


    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = total_batch_size_per_gpu

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.output_dir,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.eval_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    # unified prompting for show-o
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sod|>", "<|eod|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2d|>", "<|act|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    print('special tokens : \n', uni_prompting.sptids_dict)
    # Initialize model
    if config.model.showo.load_from_showo:
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(accelerator.device)
        if config.model.showo.vocab_size != model.vocab_size:
            model.showo.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size
            model.config.mask_token_id = model.config.vocab_size - 1
            model.mask_token_id = model.config.vocab_size - 1
    else:
        model = Showo(**config.model.showo).to(accelerator.device)
    #embedding expand
    if config.model.showo.dynamic_size:
        dynamic_size = config.model.showo.dynamic_size
        model.resize_dynamic_size(dynamic_size, 'sft', config)
        evaluator = Evaluator(config.model.eval.i3d_path, max_batchsize=config.training.batch_size_val_nus)
        evaluator.eval()
        evaluator.requires_grad_(False)
        vq_name = get_vq_model_class(config.model.vq_model.type)
        vq_model = vq_name(config_exps=config,
                           num_vq_embeddings=config.model.vq_model.num_vq_embeddings,
                           num_dyn_embeddings=config.model.vq_model.num_dyn_embeddings)

        if config.model.vq_model.get("pretrained_model_path", None):
            from safetensors.torch import load_file
            state_dict = load_file(config.model.vq_model.pretrained_model_path)  # ['model']
            vq_model.load_state_dict(state_dict, strict=True)

        vq_model.eval()
        vq_model.requires_grad_(False)

    if config.model.showo.resume_from_pretrain:
        logger.info('load video pretrain from {}'.format(config.model.showo.resume_from_pretrain))
        weights = torch.load(config.model.showo.resume_from_pretrain)
        check_sate = model.load_state_dict(weights, strict=False)

    ##################################
    #   Optimizer and LR scheduler   #
    ##################################
    optimizer_config = config.optimizer.params
    if config.training.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=32,
            lora_alpha=32*2,
            target_modules=find_all_linear_names(model),
            lora_dropout=0.01,
            bias= "none",
            task_type="CAUSAL_LM",
        )
        print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )
    logger.info("Creating dataloaders and lr_scheduler")
    dataset_config = config.dataset.params
    ####################
    #     Dataloader   #
    ####################

    if config.dataset.dataset_use == "sft_navsim":
        total_batch_size_without_accum = config.training.batch_size_train_nus * accelerator.num_processes
        total_batch_size = (total_batch_size_without_accum * config.training.gradient_accumulation_steps)
        # DatasetNavsim, DataCollatorForNavsim, DataCollatorForNavsimtest
        dataset_navsim_train = DatasetNavsim(config=config, split='train', aug_enable=False)
        dataset_navsim_val = DatasetNavsim(config=config, split='test', aug_enable=False)
        print('process index : ',
          accelerator.process_index, ', total_gpus:', accelerator.num_processes,
          "Length of dataset_train:", len(dataset_navsim_train), "samples",
          "Length of dataset_val:", len(dataset_navsim_val), "samples")

        if accelerator.num_processes > 1:
            sampler_nusc = DistributedSampler(dataset_navsim_train,
                                         num_replicas=accelerator.num_processes,
                                         rank=accelerator.process_index,
                                         shuffle=True,
                                         seed=config.training.seed
                                         )
            sampler_nusc_val = DistributedSampler(dataset_navsim_val,
                                         num_replicas=accelerator.num_processes,
                                         rank=accelerator.process_index,
                                         shuffle=False,
                                         seed=config.training.seed
                                         )
            shuffle_train = False
            shuffle_val = False
        else:
            sampler_nusc = None
            sampler_nusc_val = None
            shuffle_train = True
            shuffle_val = False
        train_dataloader_navsim = DataLoader(dataset_navsim_train, batch_size=config.training.batch_size_train_nus,
                                          sampler=sampler_nusc, collate_fn=dataset_navsim_train.collate_fn,
                                          shuffle=shuffle_train, num_workers=dataset_config.num_workers)
        val_dataloader_navsim = DataLoader(dataset_navsim_val, batch_size=config.training.batch_size_val_nus,
                                          sampler=sampler_nusc_val, collate_fn=dataset_navsim_val.collate_fn,
                                          shuffle=shuffle_val, num_workers=dataset_config.num_workers)
        num_update_steps_per_epoch = math.ceil(len(dataset_navsim_train) / total_batch_size)
        num_train_epochs_t2d = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    else:
        raise ValueError(f"Unsupported dataset")

    ####################
    #     Load ckpt    #
    ####################
    global_step = 0
    first_epoch = 0

    if config.experiment.eval_from_checkpoint and config.experiment.eval_only:
        path = config.experiment.eval.eval_dir
        logger.info(f"only evaluation from ckpt:{path}")
        if path is not None:

            accelerator.print(f"Resuming from checkpoint {path}")
            state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            del state_dict
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    vq_model = vq_model.to(accelerator.device)
    evaluator = evaluator.to(accelerator.device)
    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.model.embed_tokens.weight.dtype


    ####################
    #     Training     #
    ####################

    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    @torch.no_grad()
    def prepare_inputs_and_labels(
            prev_img_context_1s: Union[torch.FloatTensor, torch.LongTensor],
            prev_img_dynamic_1s: Union[torch.FloatTensor, torch.LongTensor],
            prev_img_context_2s: Union[torch.FloatTensor, torch.LongTensor],
            prev_img_dynamic_2s: Union[torch.FloatTensor, torch.LongTensor],
            next_img_context: Union[torch.FloatTensor, torch.LongTensor],
            next_img_dynamic: Union[torch.FloatTensor, torch.LongTensor],
            ego_status: Union[torch.FloatTensor, torch.LongTensor],
            future_trajectories: Union[torch.FloatTensor, torch.LongTensor],
            condition_len:None,
            mode = "navsim",
            is_train: bool = True,

    ):
        if mode in ["navsim"]:
            #1s
            input_ids_prev1s, labels_prev1s = vq_model.tokenize(prev_img_dynamic_1s,
                                                  context_pixel_values=prev_img_context_1s,
                                                  context_length=condition_len,
                                                  special_token=uni_prompting.sptids_dict,
                                                  return_label=False)# (batch*T,3,H,W)
            #2s
            input_ids_prev2s, labels_prev2s = vq_model.tokenize(prev_img_dynamic_2s,
                                                  context_pixel_values=prev_img_context_2s,
                                                  context_length=condition_len,
                                                  special_token=uni_prompting.sptids_dict,
                                                  return_label=False)# (batch*T,3,H,W)
            #next
            input_ids_next, labels_next = vq_model.tokenize(next_img_dynamic,
                                                  context_pixel_values=next_img_context,
                                                  context_length=condition_len,
                                                  special_token=uni_prompting.sptids_dict) # (batch*T,3,H,W)

            labels_prev1s = {key_label: torch.ones_like(labels_prev1s[key_label])*-100 for key_label in labels_prev1s}
            labels_prev2s = {key_label: torch.ones_like(labels_prev2s[key_label])*-100 for key_label in labels_prev2s}
            #1s
            for input_ids_prev, labels_prev in [(input_ids_prev1s, labels_prev1s), (input_ids_prev2s, labels_prev2s)]:
                vocab_offset = len(uni_prompting.text_tokenizer)#add offset
                for k, v in input_ids_prev.items():#context and dynamic
                    mask = (v > 0) & (v < (vocab_offset - len(uni_prompting.sptids_dict)))
                    input_ids_prev[k][mask] += vocab_offset
                    if k in labels_prev:
                        mask_label = (labels_prev[k] > 0) & (labels_prev[k] < (vocab_offset - len(uni_prompting.sptids_dict)))
                        labels_prev[k][mask_label] += vocab_offset




            for k, v in input_ids_next.items():#context and dynamic
                mask = (v > 0) & (v < (vocab_offset - len(uni_prompting.sptids_dict)))
                input_ids_next[k][mask] += vocab_offset
                if k in labels_next:
                    mask_label = (labels_next[k] > 0) & (labels_next[k] < (vocab_offset - len(uni_prompting.sptids_dict)))
                    labels_next[k][mask_label] += vocab_offset

            #caption part:
            action_num = future_trajectories.shape[1]
            texts = dict(input_caption=[""]*future_trajectories.shape[0])
            input_ids_prev = [input_ids_prev1s, input_ids_prev2s]
            labels_prev = [labels_prev1s, labels_prev2s]
            final_input_ids, labels = uni_prompting((texts, input_ids_prev, input_ids_next, labels_prev, labels_next, action_num, ego_status), mode)
        else:
            raise NotImplementedError
        return final_input_ids, labels, [input_ids_prev, input_ids_next]


    name = [n for n, p in model.named_parameters() if p.requires_grad]
    num_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    # print("Parameters require gradients: {}".format(name))
    print("Num of Parameters require gradients: {}M".format(num_params / 1e6))
    if accelerator.mixed_precision == "fp16":
        images_feat_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        images_feat_dtype = torch.bfloat16
    else:
        images_feat_dtype = torch.float32
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    min_mean_fvd = 1000
    if config.experiment.eval_only:
        eval_logs = evaluate(model,
                             vq_model,
                             config,
                             mask_dtype,
                             accelerator,
                             global_step,
                             uni_prompting,
                             val_dataloader_navsim,
                             evaluator,
                             prepare_inputs_and_labels)
    return eval_logs
    num_train_epochs = num_update_steps_per_epoch
    cur_epoch = 0
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_dataloader_navsim):
            batch_size = batch["next_img_context"].shape[0]
            (prev_img_context_1s, prev_img_dynamic_1s, prev_img_context_2s,
             prev_img_dynamic_2s, next_img_context, next_img_dynamic, token,
             ego_status, future_trajectories) = (
                batch['prev_img_context_1s'].to(accelerator.device, non_blocking=True),
                batch['prev_img_dynamic_1s'].to(accelerator.device, non_blocking=True),
                batch['prev_img_context_2s'].to(accelerator.device, non_blocking=True),
                batch['prev_img_dynamic_2s'].to(accelerator.device, non_blocking=True),
                batch['next_img_context'].to(accelerator.device, non_blocking=True),
                batch['next_img_dynamic'].to(accelerator.device, non_blocking=True),
                # batch['model_obs_140o'].to(accelerator.device, non_blocking=True),
                batch['token'],
                batch['ego_status'].to(accelerator.device, non_blocking=True) if config.experiment.add_ego else None,
                batch['future_trajectory'].to(accelerator.device, non_blocking=True),
            )
            #单独处理ego_status and H_cmd
            data_time_m.update(time.time() - end)
            with (torch.no_grad()):
                # *-------*-------*-------*-------*-------*
                # Build formatted sequences for navsim
                # *-------*-------*-------*-------*-------*
                context_length = config.dataset.ctd.context_length
                input_ids, labels, image_tokens_ori = prepare_inputs_and_labels(#model_obs_140o=model_obs_140o,
                                                                                prev_img_context_1s=prev_img_context_1s,
                                                                                prev_img_dynamic_1s=prev_img_dynamic_1s,
                                                                                prev_img_context_2s=prev_img_context_2s,
                                                                                prev_img_dynamic_2s=prev_img_dynamic_2s,
                                                                                next_img_context=next_img_context,
                                                                                next_img_dynamic=next_img_dynamic,
                                                                                ego_status=ego_status,
                                                                                future_trajectories=future_trajectories,
                                                                                condition_len=context_length)

                len_seq = input_ids.shape[-1]
                attention_mask = create_attention_mask_for_nusc(input_ids, # (B,1,L,L)
                                                                   pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                   soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                   eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                   sod_id=int(uni_prompting.sptids_dict['<|sod|>']),
                                                                   eod_id=int(uni_prompting.sptids_dict['<|eod|>']),
                                                                   rm_pad_in_image=True,
                                                                   return_inverse_mask=True,
                                                                   mask_future_ratio=None)

                attention_mask = attention_mask.to(mask_dtype)

            action_len = future_trajectories.shape[1]
            with (accelerator.accumulate(model)):#uni_prompting.sptids_dict as input
                logits, loss_video, loss_tj, mmu_index, eod_img_d = model.navsim_forward(
                    inputs=input_ids,
                    input_embeddings=None,
                    attention_mask=attention_mask,
                    labels=labels,
                    batch_size=batch_size,
                    action_len=action_len,
                    sptids_dict=uni_prompting.sptids_dict,
                    gt_tj=future_trajectories,#only training (x,y)
                    motion_weight=config.training.motion_weight,
                    ego_status=ego_status,
                    nfp_coffe=config.experiment.nfp_loss,
                )
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss_video = accelerator.gather(loss_video.repeat(batch_size)).mean()
                avg_loss_tj = accelerator.gather(loss_tj.repeat(batch_size)).mean()
                # logger.info(f"step_loss_video:{loss_video.item()}")
                loss = config.training.video_coeff * loss_video + config.training.tj_coeff * loss_tj

                accelerator.backward(loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()
                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        # "step_loss_qa": avg_loss_qa.item(),
                        "step_loss_video": avg_loss_video.item(),
                        "step_loss_tj": avg_loss_tj.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        # "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"epoch: {epoch} "
                        f"Step: {global_step + 1} "
                        f"Loss_video: {avg_loss_video.item():0.4f} "
                        f"Loss_tj: {avg_loss_tj.item():0.4f} "

                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if accelerator.is_main_process and cur_epoch!=epoch and epoch>=config.training.save_start_epoch:
                    save_checkpoint(model, config, accelerator, global_step + 1)
                if cur_epoch != epoch and (epoch>=config.training.eval_start_epoch or epoch==5):# and  ((global_step) % config.experiment.eval_every == 0 and global_step>10) or
                    if accelerator.num_processes > 1:
                        accelerator.wait_for_everyone()# every GPU generates one short video

                    eval_logs = evaluate(model,
                                        vq_model,
                                        config,
                                        mask_dtype,
                                        accelerator,
                                        global_step,
                                        uni_prompting,
                                        val_dataloader_navsim,
                                        evaluator,
                                        prepare_inputs_and_labels)
                cur_epoch = epoch
                global_step += 1
            if global_step >= config.training.max_train_steps:
                break

        if global_step >= config.training.max_train_steps:
            break
            # End for
    accelerator.wait_for_everyone()
    accelerator.end_training()

@torch.no_grad()
def visualize_predictions(model,
                        vq_model,
                        uni_prompting,
                        config,
                        global_step,
                        input_ids, #all task tokenized GT
                        logits, #all task predicted
                        image_ori, #t2d tokenized context and dynamic
                        token_ori, #t2d GT pixel
                        token,
                        accelerator,
                        num_per_frame=30,
                        pred_frames=10
                            ):
    logger.info("Visualizing training set logits...")
    model.eval()
    batch_ids = torch.randint(low=0, high=input_ids.shape[0], size=(1,)).to(input_ids.device)
    if accelerator.is_main_process:
        with torch.no_grad():
            prev_token, next_token = token_ori
            #pixel value images
            images_prev = torch.clamp((image_ori[1][batch_ids] + 1.0) / 2.0, min=0.0, max=1.0)[0]  # (T(c+d), 3, H, W)
            images_prev *= 255.0
            images_next = torch.clamp((image_ori[3][batch_ids] + 1.0) / 2.0, min=0.0, max=1.0)[0]  # (T(c+d), 3, H, W)
            images_next *= 255.0
            images = torch.cat((images_prev[1:], images_next[1:]), dim=0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


            #prev images  GT
            context_len_prev = len(prev_token['context'][0])//450
            #Tokenizer recon
            recons_images_prev, recons_lens_prev = vq_model.detokenize(indices=prev_token,
                                                                       batch_ids=batch_ids,
                                                                       offset_tokenzier=len(uni_prompting.text_tokenizer),
                                                                       sptids_dict=uni_prompting.sptids_dict)#(B, T,C,H,W) tokenizer reconstruct
            recons_images_prev = torch.clamp((recons_images_prev + 1.0) / 2.0, min=0.0, max=1.0)
            recons_images_prev = 255.0*recons_images_prev
            recons_images_prev = recons_images_prev[0]
            recons_images_next, recons_lens_next = vq_model.detokenize(indices=next_token,
                                                                       batch_ids=batch_ids,
                                                                       offset_tokenzier=len(uni_prompting.text_tokenizer),
                                                                       sptids_dict=uni_prompting.sptids_dict)#(B, T,C,H,W) tokenizer reconstruct
            recons_images_next = torch.clamp((recons_images_next + 1.0) / 2.0, min=0.0, max=1.0)
            recons_images_next = 255.0*recons_images_next
            recons_images_next = recons_images_next[0]

            recons_images = torch.cat((recons_images_prev[2:], recons_images_next[2:]), dim=0).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

            # nusc Predicted recon
            pred_logits_token = dict(context=next_token["context"], dynamic=logits.argmax(-1))
            d_start_sod = torch.arange(0, pred_frames)*num_per_frame
            d_end_sod = torch.arange(1, pred_frames+1)*num_per_frame-1
            predicted_images, _ = vq_model.detokenize(pred_logits_token,
                                                   batch_ids=batch_ids,
                                                   offset_tokenzier=len(uni_prompting.text_tokenizer),
                                                   sptids_dict=uni_prompting.sptids_dict,
                                                   c_start_sod=None,
                                                   c_end_sod=None,
                                                   d_start_sod=d_start_sod.to(input_ids.device),
                                                   d_end_sod=d_end_sod.to(input_ids.device),
                                                   )#(T-1,C,H,W)
            predicted_images = torch.clamp((predicted_images + 1.0) / 2.0, min=0.0, max=1.0)
            predicted_images *= 255.0
            predicted_images = predicted_images[0][2:].permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            t2d_v = video_concate(images, recons_images, predicted_images)

            t2d_v = np.stack(t2d_v, 0)
            output_root = os.path.join(config.experiment.output_dir, "visual_gif", "train")
            os.makedirs(output_root, exist_ok=True)
            imageio.mimsave(os.path.join(output_root, f"{global_step}_{token[batch_ids]}.gif"), t2d_v, fps=10, loop=0)
            t2d_v = np.transpose(t2d_v, (0, 3, 1, 2))
            # video display
            wandb.log({"Original v.s. Reconstructed v.s. Predicted": wandb.Video(t2d_v, fps=10, format="webm", caption=token[batch_ids])}, step=global_step)

        logger.info("Visualizing finished...")


def save_as_webm(t2d_v, output_path, fps=10):
    T, C, H, W = t2d_v.shape
    assert C in (1, 3), "The channel dimension (C) must be 1 (grayscale) or 3 (RGB)."
    frames = [t2d_v[i].transpose(1, 2, 0) for i in range(T)]
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec="libvpx", verbose=False)


def process_images(accelerator, images, evaluator, detector_kwargs, max_decode_batchsize=20):

    images = images.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, c, t, h, w]
    if max_decode_batchsize is not None and images.shape[0] > max_decode_batchsize:
        features = batch_forward(
            max_decode_batchsize,
            images * 255.,
            lambda x: accelerator.unwrap_model(evaluator).i3d_model(x, **detector_kwargs)
        )
    else:
        features = accelerator.unwrap_model(evaluator).i3d_model(images * 255., **detector_kwargs)
    gathered_features = accelerator.gather(features)
    return gathered_features

def video_metrics_process(config, mse_values, fvd, psnr_values, ssim_values, lpips_values):

    eval_logs = {
        'eval/mse': torch.cat(mse_values, 0).mean().item(),
    }
    if config.experiment.eval.use_fvd:
        eval_logs.update({'eval/fvd': fvd})
    if config.experiment.eval.use_frame_metrics:
        eval_logs.update({
            'eval/psnr': torch.cat(psnr_values, 0).mean().item(),
            'eval/ssim': torch.cat(ssim_values, 0).mean().item(),
            'eval/lpips': torch.cat(lpips_values, 0).mean().item(),
        })
    return eval_logs

@torch.no_grad
def evaluate(model,
             vq_model,
             config,
             mask_dtype,
             accelerator,
             global_step,
             uni_prompting,
             eval_dataloader,
             evaluator,
             prepare_inputs_and_labels):

    model.eval()
    losses = []
    future_seconds = 4
    scoring_params = OmegaConf.load(config.dataset.scoring_path)
    cache_path = os.path.join(config.experiment.base_root, 'dataset/navsim/cache', config.dataset.scene_filter.test_cache)
    score_rows = []
    mse_values, psnr_values, ssim_values, lpips_values, fvds, desc_a_pair, action_pair = [], [], [], [], [], [], []
    real_feats, gen_feats = FeatureStats(capture_mean_cov=True), FeatureStats(capture_mean_cov=True)
    eval_iters = min(len(eval_dataloader), config.experiment.eval.max_eval_iters)
    bar = tqdm(range(eval_iters), desc="validation", disable=not accelerator.is_local_main_process)
    logger.info("validation PDMS...")
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
    num_visual = 0
    dynamic_tok_num = 30
    output_root = config.experiment.output_dir
    os.makedirs(output_root, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):

            if i == config.experiment.eval.max_eval_iters:
                break

            batch_size = batch["next_img_context"].shape[0]
            (prev_img_context_1s, prev_img_dynamic_1s, prev_img_context_2s,
             prev_img_dynamic_2s, next_img_context, next_img_dynamic, token,
             ego_status) = (
                batch['prev_img_context_1s'].to(accelerator.device, non_blocking=True),
                batch['prev_img_dynamic_1s'].to(accelerator.device, non_blocking=True),
                batch['prev_img_context_2s'].to(accelerator.device, non_blocking=True),
                batch['prev_img_dynamic_2s'].to(accelerator.device, non_blocking=True),
                batch['next_img_context'].to(accelerator.device, non_blocking=True),
                batch['next_img_dynamic'].to(accelerator.device, non_blocking=True),
                batch['token'],
                batch['ego_status'].to(accelerator.device, non_blocking=True) if config.experiment.add_ego else None,
                # batch['future_trajectory'].to(accelerator.device, non_blocking=True),
            )
            token_encode = torch.tensor([tok.encode("utf-8") for tok in token], device=accelerator.device, dtype=torch.uint8)
            future_trajectories = torch.zeros_like(batch['future_trajectory'], device=accelerator.device, dtype=batch['future_trajectory'].dtype)
            with (torch.no_grad()):
                # Encode images to image tokens, mask them and create input and labels
                context_length = config.dataset.ctd.context_length
                input_ids, labels, image_tokens_ori = prepare_inputs_and_labels(prev_img_context_1s=prev_img_context_1s,
                                                                                prev_img_dynamic_1s=prev_img_dynamic_1s,
                                                                                prev_img_context_2s=prev_img_context_2s,
                                                                                prev_img_dynamic_2s=prev_img_dynamic_2s,
                                                                                next_img_context=next_img_context,
                                                                                next_img_dynamic=next_img_dynamic,
                                                                                ego_status=ego_status,
                                                                                future_trajectories=future_trajectories,
                                                                                condition_len=context_length)#pad:50295

                len_seq = input_ids.shape[-1]
                attention_mask = create_attention_mask_for_nusc(input_ids, # (B,1,L,L)
                                                                   pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                   soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                   eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                   sod_id=int(uni_prompting.sptids_dict['<|sod|>']),
                                                                   eod_id=int(uni_prompting.sptids_dict['<|eod|>']),
                                                                   rm_pad_in_image=True,
                                                                   return_inverse_mask=True,
                                                                   mask_future_ratio=None)

                attention_mask = attention_mask.to(mask_dtype)

            sod_input_idx = torch.where(input_ids == uni_prompting.sptids_dict['<|sod|>'].to(input_ids.device))[1].unique()
            eod_input_idx = torch.where(input_ids == uni_prompting.sptids_dict['<|eod|>'].to(input_ids.device))[1].unique()
            eot_input_idx = torch.where(input_ids == uni_prompting.sptids_dict['<|eot|>'].to(input_ids.device))[1].unique()
            action_len = future_trajectories.shape[1]
            #add cmd token
            input_embed = model.showo.model.embed_tokens(input_ids[:, :eot_input_idx[-1]+dynamic_tok_num+1])
            if ego_status is not None:
                ego_token = model.ego_forward(ego_status.to(input_embed.dtype))
                input_embed[:, eot_input_idx[0]-1, :] = ego_token
            input_attention_mask = attention_mask[:, :, :eot_input_idx[-1]+dynamic_tok_num+1, :eot_input_idx[-1]+dynamic_tok_num+1]
            time_0 = time.time()
            with torch.autocast("cuda", dtype=torch.float32, enabled=accelerator.mixed_precision != "no"):
                gen_image_token_ids, gen_trj = accelerator.unwrap_model(model).navsim_gen(input_embed=input_embed,
                                            attention_mask=input_attention_mask,
                                            config=config,
                                            action_len=action_len,
                                            uni_prompting=uni_prompting,
                                            ego_status=ego_status,
                                            )

            #####################################
            # ---------video metrics------------#
            #####################################
            infer_time = time.time() - time_0
            logger.info(f'infer_time:{infer_time}s')
            predicted_images = img_token2pixel(image_tokens_ori[1], uni_prompting, vq_model, gen_image_token_ids)[:, 2:]
            recons_images = img_token2pixel(image_tokens_ori[1], uni_prompting, vq_model)[:, 2:]
            pixel_values = torch.clamp((next_img_dynamic + 1.0) / 2.0, min=0.0, max=1.0)[:, 1:]

            if config.experiment.eval.use_fvd:
                with torch.autocast("cuda", dtype=torch.float32, enabled=False):
                    detector_kwargs = dict(rescale=True, resize=True, return_features=True)
                    real_feat = process_images(accelerator, pixel_values, evaluator, detector_kwargs)
                    gen_feat = process_images(accelerator, predicted_images, evaluator, detector_kwargs)
                    if accelerator.num_processes > 1:
                        accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        real_feats.append_torch(real_feat)
                        gen_feats.append_torch(gen_feat)
                        fvd = accelerator.unwrap_model(evaluator).compute_fvd(real_feats, gen_feats)
                        fvds.append(torch.tensor(fvd).repeat(batch_size))
                        logger.info(f"current fvd estimate:{fvd}")
            # #video metrics:mse,psnr,ssim,lpips
            if config.experiment.eval.use_frame_metrics:
                with torch.autocast("cuda", dtype=torch.float32, enabled=False):
                    mse_value, psnr_value, ssim_value, lpips_value = accelerator.unwrap_model(evaluator)(pixel_values.clamp(0.0, 1.0), predicted_images)
                    # mse_value, psnr_value, ssim_value, lpips_value = accelerator.unwrap_model(evaluator)(pixel_values.clamp(0.0, 1.0)[:,:-1], predicted_images[:, 2:])
                    mse_values.append(accelerator.gather(mse_value.repeat(batch_size)))
                    psnr_values.append(accelerator.gather(psnr_value.repeat(batch_size)))
                    ssim_values.append(accelerator.gather(ssim_value.repeat(batch_size)))
                    lpips_values.append(accelerator.gather(lpips_value.repeat(batch_size)))

            #####################################
            # ---------text  metrics------------#
            #####################################
            #none
            #####################################
            # ---------action metrics-----------#
            #####################################
            # action metrics
            if config.experiment.eval.use_trj_metrics:
                samples_eval = []
                for trj_i in range(gen_trj.shape[0]):
                    gathered_gen_trj = accelerator.gather(gen_trj[trj_i][None, ...].contiguous())
                    gathered_token = accelerator.gather(token_encode[trj_i][None, ...].contiguous())
                    assert gathered_gen_trj.shape[0] == gathered_token.shape[0], ( f"Shape mismatch: gathered_gen_trj has {gathered_gen_trj.shape[0]} samples, "
                                                                                   f"but gathered_token has {gathered_token.shape[0]} samples." )
                    # print(f"gathered_token gather from {gathered_token.shape[0]} gpus ....")
                    for gather_idx in range(gathered_gen_trj.shape[0]):
                        result_action = dict(cfg=scoring_params,
                                         cache_path=Path(cache_path),
                                         token=bytes(gathered_token[gather_idx].cpu().numpy().tolist()).decode("utf-8"),
                                         future_trajectory=gathered_gen_trj[gather_idx].detach().cpu().to(torch.float32))
                        samples_eval.append(result_action)
                if accelerator.is_main_process:
                    score_rows.extend(PDSM_eval(config, samples_eval, logger))

            # video display
            if accelerator.is_main_process and num_visual <= 80:
                magic_number = np.random.rand()
                if magic_number < 0.45:
                    b_sample = np.random.randint(batch_size)
                    num_visual+=1
                    # if i == 11:
                    #     print("debug here")
                    o_images = (255.0*pixel_values[b_sample]).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                    r_images = (255.0*recons_images[b_sample]).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                    p_images = (255.0*predicted_images[b_sample]).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

                    t2d_v = video_concate(o_images, r_images, p_images, context_length)
                    t2d_v = np.stack(t2d_v, 0)
                    val_root = os.path.join(output_root, "visual_gif", "val")
                    os.makedirs(val_root, exist_ok=True)
                    imageio.mimsave(os.path.join(val_root,
                                                 f"step_{global_step}_{token[b_sample]}.gif"), t2d_v, fps=10, loop=0)
                    t2d_v = np.transpose(t2d_v, (0, 3, 1, 2))
                    final_step = global_step+i
                    if num_visual <= 10:
                        wandb.log({"VAL-Original v.s. Reconstructed v.s. generated": wandb.Video(t2d_v, fps=10, format="webm",
                                   caption=token[b_sample])}, step=final_step)
                else:
                    pass
            bar.update(1)
    if accelerator.is_main_process:
        #video val
        eval_logs = video_metrics_process(config, mse_values, fvd, psnr_values, ssim_values, lpips_values)
        accelerator.log(eval_logs, step=global_step+i)

        #traj test
        trj_logs = pdsm_score_process(config, score_rows, global_step, logger)
        trj_logs = {f"PDMS/{k}":v for k,v in trj_logs.items() if k not in ['token','valid']}
        accelerator.log(trj_logs, step=global_step+i)

    if accelerator.num_processes > 1:
        accelerator.wait_for_everyone()

    model.train()
    logger.info("validation finished...")
    if accelerator.is_main_process:
        return eval_logs
    else:
        return None

def save_checkpoint(model, config, accelerator, global_step,min_step=None):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)
    if min_step:
        save_path = Path(output_dir) / f"checkpoint_w_min_fvd_{global_step}"
    else:
        save_path = Path(output_dir) / f"checkpoint_step_{global_step}"
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":

    main()
