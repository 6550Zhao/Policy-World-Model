
import copy
from dataclasses import dataclass, field
import json
from torch import Tensor
from datasets import load_dataset, load_from_disk
import numpy as np
from data_utils.dataset_config import _get_rawvideo_dec,process_coco_image
from data_utils.dataset_config import OPENDV_LOCAL, OPENDV_MINI, OPENDV_FULL, NUSCENES_FRONT, NUSCENES_BACK, NUSCENES_FRONT_LEFT, NUSCENES_FRONT_RIGHT, NUSCENES_BACK_LEFT, NUSCENES_BACK_RIGHT
import os
from training.utils import get_config, flatten_omega_conf, image_transform
from PIL import Image
from llava.llava import conversation as conversation_lib
from nuscenes.utils.splits import create_splits_scenes
import math
from PIL import UnidentifiedImageError
local_rank = None
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union
import os
from navsim.common.dataclasses import SensorConfig, Scene, Trajectory, NAVSIM_INTERVAL_LENGTH
from hydra.utils import instantiate
from pathlib import Path
import lzma
import pickle
import os
import torch
import logging
from dataclasses import dataclass, field, asdict
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
DataConfig = {
    "OPENDV_LOCAL": [OPENDV_LOCAL],
    "OPENDV_MINI": [OPENDV_MINI],
    "OPENDV_FULL": [OPENDV_FULL],
}
prompt = {
    "generate_scene": "Draw image in front of a vehicle.",
    "desc": "Summarize what the image shows.",
    "action": "What should be the next move?",
    "plan": "Give the next 6 plans.",
    "image": "Draw the image of the next frame."
}
DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."

def preprocess_multimodal(sources):
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = sentence['value'].strip()

    return sources

def preprocess_v0(
        sources,
        tokenizer,
        return_system = False,
        max_len=None,
):
    has_image = False
    conv = conversation_lib.default_conversation.copy()
    roles = {"USER": conv.roles[0], "ASSISTANT": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversation_str = str(conv.get_prompt()).strip()
        conversations.append(conversation_str)
    if max_len is not None:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=max_len,
            truncation=True,
        ).input_ids
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "                   # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):        # loop for instances in a batch
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv.sep2)  # in phi-2, pad_token_id == eos_token_id
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        pad_len = sum(target==tokenizer.pad_token_id)
        rounds = conversation.split(conv.sep2)              # handle multi-round conversation regarding one image
        cur_len = pad_len                                         # no bos token in phi, so set the initial len to 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX#usr->-100


    input_ids_system = tokenizer(
        [SYSTEM_PROMPT for _ in range(len(conversations))],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    if return_system == True:

        return dict(
            input_ids=input_ids,
            labels=targets,
            input_ids_system=input_ids_system
        )
    else:
        return dict(input_ids=input_ids,
            labels=targets,)
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    # tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_text = [instance["input_text"] for instance in instances]
        batch = dict(
            input_text=input_text,
        )
        for i in ['image_clip', 'image_vq', 'action']:
            if i in instances[0]:
                state_action = [instance[i] for instance in instances]

                new_images = []
                for image in state_action:
                    if type(image) is list:
                        for i in image:
                            new_images.append(i)
                    else:
                        new_images.append(image)
                images = new_images

                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch[i] = torch.stack(images)
                else:
                    batch[i] = images
                #image_vq(batch,T,C,H,W)
        return batch

class DatasetNuScenes(Dataset):#camera ready dataset
    def __init__(self,
                 config,
                 split,
                 version='v1.0-trainval',
                 aug_enable=False,
                 scene_split=False,
                 scene_name=None,
                 aug= {
                        'brightness': [0.9, 1.1],
                        'contrast': [0.9, 1.1],
                        'saturation': [0.9, 1.1],
                        'hue': [-0.05, 0.05],
                        'random_resized_crop_scale': (0.9, 1.0),
                        'random_resized_crop_ratio': (0.5, 0.6),

                        },
                 ):
        super(DatasetNuScenes, self).__init__()
        assert config.dataset.ctd.nuscenes_data_path is not None, "Either nusc or nuscenes_data_path"
        # with open(os.path.join(config.dataset.ctd.anno_path, f'nuscenes2d_ego_temporal_infos_{split}.pkl'), 'rb') as f:
        #     self.nus_ori_annos = pickle.load(f)['infos']
        # self.omini_anno_root = config.dataset.ctd.anno_path  # [[conv[qas]]]
        self.scenes = create_splits_scenes()
        with open(os.path.join(config.dataset.ctd.image_file, f'CAM_FRONT_{split}_imgs_path.json'), 'r')as f:
            self.image_path = json.load(f)
        # self.image_root = config.dataset.ctd.image_root
        if split == 'train':
            with open(os.path.join(config.dataset.ctd.image_file, 'ominidrive', f'plan_{split}_filter_w_ego_w_cmd_1s_to_19s.json'), 'r') as f:
                self.omini_annos = json.load(f)
            self.scenes = self.scenes['train']
        elif split == 'val':
            if scene_split:
                with open(os.path.join(config.dataset.ctd.image_file, 'ominidrive', 'val_saparate_scene/val_1s_20s.json'), 'r') as f:
                    omini_annos = json.load(f)
                assert scene_name is not None, "require scene name"
                self.omini_annos = omini_annos[scene_name]
            else:
                # with open(os.path.join(config.dataset.ctd.omini_path, f'plan_{split}_filter_w_ego_w_cmd_1s_to_20s.json'), 'r') as f:
                with open(os.path.join(config.dataset.ctd.image_file, 'ominidrive', f'plan_{split}_filter_w_ego_w_cmd_1s_to_19s.json'), 'r') as f:
                    self.omini_annos = json.load(f)
            self.scenes = self.scenes['val']
        else:
            raise NotImplementedError
        self.camera_views= [k for k, v in config.dataset.ctd.views.items() if v is not '']
        self.num_images = config.dataset.ctd.segment_length#过去2s，那么就是2x8=16帧 frames condition
        self.condition_frames = config.dataset.ctd.condition_length #2 for one second
        # self.data = self._prepare_text_vqa()
        # random.shuffle(self.data)#字典，image file
        self.vq_processor = image_transform
        self.Con_resolution_h, self.Con_resolution_w = config.dataset.ctd.c_resolution
        self.resolution_h, self.resolution_w = config.dataset.ctd.d_resolution
        self.prev_frames = config.dataset.ctd.prev_frames  # 12 #1s
        self.next_frames = config.dataset.ctd.next_frames
        self.aug_enable = aug_enable
        self.aug = aug
        self.split = split
        self.collate_fn = DataCollatorForSupervisedNuScenes()
        self.fps = config.dataset.ctd.next_frames #12
    def _check_frame_images(self, frame_token):
        sample = self.nusc.get('sample', frame_token)
        for _ in range(self.num_images-1):
            if sample['next'] == '':
                return False
            sample = self.nusc.get('sample', sample['next'])
        return True
    def data_augmentation(self, images):

        con_len_count = 0
        new_images_context = []
        new_images = []
        tensor = transforms.ToTensor()
        for image_0 in images:
            if con_len_count in [0, 1]:
                image = transforms.Resize((self.Con_resolution_h, self.Con_resolution_w), interpolation=transforms.InterpolationMode.BICUBIC)(image_0)
                image = transforms.CenterCrop((self.Con_resolution_h, self.Con_resolution_w))(image)
                image = tensor(image) #/255
                image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
                new_images_context.append(image)

            image = transforms.Resize((self.resolution_h, self.resolution_w), interpolation=transforms.InterpolationMode.BICUBIC)(image_0)
            image = transforms.CenterCrop((self.resolution_h, self.resolution_w))(image)
            image = tensor(image) #/255
            image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
            new_images.append(image)
            con_len_count += 1
        return torch.stack(new_images_context,0) if new_images_context is not [] else None, torch.stack(new_images,0) if new_images_context is not [] else None
    def get_jittor_params(self,
            brightness: Optional[List[float]],
            contrast: Optional[List[float]],
            saturation: Optional[List[float]],
            hue: Optional[List[float]],
    ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h
    def get_crop_params(self, img: Tensor, scale: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop with fixed aspect ratio.

        Args:
            img (Tensor): Input image (C, H, W).
            scale (list): Range of scale of the origin size cropped.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop, keeping the original aspect ratio.
        """
        _, height, width = F.get_dimensions(img)
        area = height * width

        original_ratio = width / height

        for _ in range(3):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            h = int(round(math.sqrt(target_area / original_ratio)))
            w = int(round(h * original_ratio))
            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w
        if width > height:
            w = int(round(height * original_ratio))
            h = height
        else:
            h = int(round(width / original_ratio))
            w = width

        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
    def get_segment(self, num_frames, start_index, end_index, select_num, short_flag=False, mode=None):
        start_index = start_index
        end_index = end_index
        if start_index >= num_frames or end_index >= num_frames:
            print(f'start_index: {start_index}, start_index: {end_index},total_frames: {num_frames}')
            raise ValueError("start_index must be less than num_frames")
        indices = np.arange(start_index, end_index)
        if mode == 'dynamic':
            weights = indices - start_index + 1
        else:
            weights = np.ones_like(indices)
        # select_num index
        if short_flag:
            selected_indices = np.random.choice(indices, size=select_num, replace=True, p=weights/weights.sum())
        else:
            selected_indices = np.random.choice(indices, size=select_num, replace=False, p=weights/weights.sum())

        return selected_indices.tolist()
    def _load_image(self, img_path):
        # Implement your image loading logic here
        # For example, using PIL:

        img = Image.open(img_path)
        # print(f"Loading image from: {img_path}")
        # img = img.resize((224, 224))  # Resize if necessary
        # img = torch.tensor(np.array(img)).permute(2, 0, 1)  # Convert to tensor and rearrange dimensions
        return img

    def __len__(self):
        return len(self.omini_annos)

    def process_desc(self,desc):
        banned = ['rear', 'Behind', 'behind', 'right-rear', 'left-rear']
        sentences = desc.split('. ')
        filtered = []
        for s in sentences:
            f = True
            for b in banned:
                if b in s:
                    f = False
                    break
            if f:
                filtered.append(s)
        new_desc = ''
        for s in filtered:
            new_desc += s
            new_desc += '. '
        new_desc = new_desc[:-2]
        return new_desc

    def _get_images(self, frame_token):
        images = {}
        images_prev_root, images_next_root = self.image_path[frame_token]['prev'], self.image_path[frame_token]['next']
        len_prev_max = self.prev_frames
        len_next_max = self.next_frames
        #prev
        prev_img = []
        next_img = []
        current_observe = []
        if len(images_prev_root) >= len_prev_max:
            for i in range(len_prev_max):
                prev_img.append(self._load_image(os.path.join(self.image_file,images_prev_root[i-len_prev_max])))
        else:
            for i in range(len(images_prev_root)):
                prev_img.append(self._load_image(os.path.join(self.image_file, images_prev_root[i - len(images_prev_root)])))
        if len(images_next_root) >= len_next_max:
            for i in range(len_next_max):
                next_img.append(self._load_image(os.path.join(self.image_file, images_next_root[i])))
        else:
            for i in range(len(images_next_root)):
                next_img.append(self._load_image(os.path.join(self.image_file, images_next_root[i])))
        return prev_img, next_img
    def command_to_text(self,command):

        if command == 2: #'FORWARD'
            text_command = 'FORWARD'
        elif command == 0: #'LEFT'
            text_command = 'LEFT'
        elif command == 1:  # 'RIGHT'
            text_command = 'RIGHT'
        else:
            raise NotImplementedError
        return text_command
    def pad_frame(self,
                  prev_img_context,
                  prev_img_dynamic,
                  next_img_context,
                  next_img_dynamic):
        t_p,c_p,h_p,w_p = prev_img_dynamic.shape
        t_n,c_n,h_n,w_n = next_img_dynamic.shape
        # if t_p < self.fps or t_n<self.fps:
        #     print("debug in here")
        if prev_img_dynamic.shape[0] == 1:
            assert prev_img_context.shape[0]==1
            prev_img_context = torch.cat([prev_img_context, prev_img_context], dim=0)#assert prev_img_context.shape[0]==2
        prev_img_dynamic = torch.cat([torch.ones((self.prev_frames-t_p,c_p,h_p,w_p), dtype=torch.float32)*-100, prev_img_dynamic], dim=0)#pad using -100

        if next_img_dynamic.shape[0] == 1:
            assert next_img_context.shape[0] == 1
            next_img_context = torch.cat([next_img_context, next_img_context], dim=0)
        next_img_dynamic = torch.cat([next_img_dynamic, torch.ones((self.next_frames-t_n,c_n,h_n,w_n), dtype=torch.float32)*-100], dim=0)#pad using -100

        return prev_img_context, prev_img_dynamic, next_img_context, next_img_dynamic, t_p, t_n
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = {}
        info = list()
        omini_anno = self.omini_annos[idx][0]
        token = omini_anno[-1]['token']

        desc_a = omini_anno[1]['a'] #description
        action_a = omini_anno[-3]['a'] #action description
        planning_a = omini_anno[-2]['plan'] #trajectory
        plan_mask = omini_anno[-2]['plan_mask']
        scene_name = omini_anno[-2]['scene_name']
        prev_img, next_img = self._get_images(token)
        images_prev_miss, images_next_miss = self.prev_frames-len(prev_img), self.next_frames-len(next_img)
        assert images_prev_miss>=0 and images_next_miss>=0

        prev_img_context, prev_img_dynamic = self.data_augmentation(prev_img)
        next_img_context, next_img_dynamic = self.data_augmentation(next_img)
        #using -100 as pad value
        prev_img_context, prev_img_dynamic, next_img_context, next_img_dynamic, t_p,t_n = self.pad_frame(prev_img_context, prev_img_dynamic, next_img_context, next_img_dynamic)
        assert [0,0] not in omini_anno[-2]['plan_mask'], print('plan_mask', omini_anno[-2]['plan_mask'])
        sample['prev_img_context'] = prev_img_context
        sample['prev_img_dynamic'] = prev_img_dynamic
        sample['next_img_context'] = next_img_context
        sample['next_img_dynamic'] = next_img_dynamic
        sample['token'] = token
        sample['desc_a'] = desc_a
        sample['action_a'] = action_a.split('\n')[0]#.split('.')[0]+'.'
        sample['idx'] = omini_anno[-2]['id']
        sample['planning_a'] = torch.tensor(planning_a, dtype=torch.float32)[0]
        sample['plan_mask'] = torch.tensor(plan_mask, dtype=torch.float32)[0]
        sample['real_len'] = torch.tensor([t_p, t_n], dtype=torch.int64)
        sample['ego_status'] = torch.tensor(omini_anno[-2]['ego_status'], dtype=torch.float32)
        sample['H_cmd'] = torch.tensor(omini_anno[-2]['plan_command'], dtype=torch.int64)
        sample['scene_name'] = scene_name
        # assert self.nus_ori_annos[omini_anno[-2]['id']]['token']==token
        return sample

class DatasetNavsim(Dataset):
    def __init__(self,
                 config,
                 split,
                 scene_filter: SceneFilter = None,
                 aug_enable=False,
                 aug= {
                        'brightness': [0.9, 1.1],
                        'contrast': [0.9, 1.1],
                        'saturation': [0.9, 1.1],
                        'hue': [-0.05, 0.05],
                        'random_resized_crop_scale': (0.9, 1.0),
                        'random_resized_crop_ratio': (0.5, 0.6),

                        },
                 ):
        super(DatasetNavsim, self).__init__()
        self.split = split #train or test
        self.navsim_root = config.dataset.navsim_root
        os.environ["NUPLAN_MAPS_ROOT"] = os.path.join(self.navsim_root, "maps")
        cache_name = config.dataset.scene_filter.train_cache if split == 'train' else config.dataset.scene_filter.test_cache
        log_name = config.dataset.logs_file.train_split if split == 'train' else config.dataset.logs_file.test_split
        filter_name = config.dataset.scene_filter.train_split if split == 'train' else config.dataset.scene_filter.test_split
        self.logs_data = os.path.join(self.navsim_root, 'navsim_logs', log_name)
        self.cache_path = os.path.join(config.experiment.base_root,'dataset/navsim/cache', cache_name)
        self.scene_filter = instantiate(OmegaConf.load(os.path.join(config.dataset.scene_filter.filter_root, f'{filter_name}.yaml')))
        self.nuplan_10hz_name = ['10hz_train','10hz_val'] if split == 'train' else ['10hz_test']
        self.metric_cache_loader = MetricCacheLoader(Path(self.cache_path))
        self.scene_filter.tokens = self.metric_cache_loader.tokens
        self.scene_loader = SceneLoader(
            sensor_blobs_path=Path(self.navsim_root,'sensor_blobs', log_name),
            data_path=Path(self.logs_data),
            scene_filter=self.scene_filter,
            sensor_config=SensorConfig.build_front_sensors(),
        )
        self._trajectory_sampling = TrajectorySampling(num_poses=config.dataset.proposal_sampling.num_poses,
                                                       time_horizon=config.dataset.proposal_sampling.time_horizon,
                                                       interval_length=config.dataset.proposal_sampling.interval_length)
        self.nuplan_10hz_blobs = os.path.join(self.navsim_root, 'nuplan_scene_blobs')
        self.nuplan_10hz_logs = config.dataset.nuplan_10hz_logs
        self.nuplan_10hz_split = ['train','val'] if split == 'train' else ['test']
        self.camera_views = [k for k, v in config.dataset.ctd.views.items() if v != '']
        self.num_images = config.dataset.ctd.segment_length  #
        self.condition_frames = config.dataset.ctd.condition_length  # 2 for one second

        self.Con_resolution_h, self.Con_resolution_w = config.dataset.ctd.c_resolution
        self.resolution_h, self.resolution_w = config.dataset.ctd.d_resolution
        self.prev_frames = config.dataset.ctd.prev_frames  # 24 #2s
        self.next_frames = config.dataset.ctd.next_frames
        self.aug_enable = aug_enable
        self.aug = aug
        # self.split = split
        self.collate_fn = DataCollatorForNavsim() # if split == 'train' else DataCollatorForNavsimtest()
        self.fps = config.dataset.ctd.fps  # 12

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=8)
        # if self.split == 'train':
        return torch.tensor(future_trajectory.poses)
        # return future_trajectory

    def get_raw_camera(self, token):
        raw_camera_path = None
        for file in self.nuplan_10hz_split:
            raw_logs_file = os.path.join(self.nuplan_10hz_logs, file, f'{token}.pkl')
            if os.path.exists(raw_logs_file):
                try:
                    with open(raw_logs_file, 'rb') as f:
                        raw_camera_path = pickle.load(f)
                except EOFError:
                    print(f"[ERROR] Failed to load pickle file (EOFError): {raw_logs_file}")
                    raise
                except Exception as e:
                    print(f"[ERROR] Unexpected error loading pickle file: {raw_logs_file}, error: {e}")
                    raise
        if raw_camera_path is not None:
            pass
        else:
            raise RuntimeError
        return raw_camera_path
    def AgentInput(self, token,
                   prev_img_context_1s,
                   prev_img_dynamic_1s,
                   prev_img_context_2s,
                   prev_img_dynamic_2s,
                   next_img_context,
                   next_img_dynamic,
                   scene_log,
                   ego_status,
                   future_trajectory):
        return dict(token=token,
                    prev_img_context_1s=prev_img_context_1s,
                    prev_img_dynamic_1s=prev_img_dynamic_1s,
                    prev_img_context_2s=prev_img_context_2s,
                    prev_img_dynamic_2s=prev_img_dynamic_2s,
                    next_img_context=next_img_context,
                    next_img_dynamic=next_img_dynamic,
                    scene_log=scene_log,
                    ego_status=ego_status,
                    future_trajectory=future_trajectory)
    def _get_images(self, token):
        images = {}
        agent_raw_camera_path = self.get_raw_camera(token)#10hz
        images_prev_info, images_next_info = [agent_raw_camera_path['past_frame_info'][view] for view in self.camera_views], [agent_raw_camera_path['future_frame_info'][view] for view in self.camera_views]
        len_prev_max = self.prev_frames
        len_next_max = self.next_frames
        #prev
        prev_img = []
        next_img = []
        for view_prev, view_next in zip(images_prev_info, images_next_info):
            if len(view_prev) >= len_prev_max:
                for i in range(len_prev_max):
                    prev_img.append(self._load_image(self.nuplan_10hz_blobs, self.nuplan_10hz_name, view_prev[i-len_prev_max]['filename_jpg']))
            else:
                for i in range(len(view_prev)):
                    prev_img.append(self._load_image(self.nuplan_10hz_blobs, self.nuplan_10hz_name, view_prev[i - len(view_prev)]['filename_jpg']))
            if len(view_next) >= len_next_max:
                for i in range(len_next_max):
                    next_img.append(self._load_image(self.nuplan_10hz_blobs, self.nuplan_10hz_name, view_next[i]['filename_jpg']))
            else:
                for i in range(len(view_next)):
                    next_img.append(self._load_image(self.nuplan_10hz_blobs, self.nuplan_10hz_name, view_next[i]['filename_jpg']))
        return prev_img, next_img

    def data_augmentation(self, images):

        con_len_count = 0
        new_images_context = []
        new_images = []
        tensor = transforms.ToTensor()
        for image_0 in images:
            if con_len_count in [0, 1]:
                image = transforms.Resize((self.Con_resolution_h, self.Con_resolution_w), interpolation=transforms.InterpolationMode.BICUBIC)(image_0)
                image = transforms.CenterCrop((self.Con_resolution_h, self.Con_resolution_w))(image)
                image = tensor(image) #/255
                image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
                new_images_context.append(image)

            image = transforms.Resize((self.resolution_h, self.resolution_w), interpolation=transforms.InterpolationMode.BICUBIC)(image_0)
            image = transforms.CenterCrop((self.resolution_h, self.resolution_w))(image)
            image = tensor(image) #/255
            image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
            new_images.append(image)
            con_len_count += 1
        return torch.stack(new_images_context,0) if new_images_context is not [] else None, torch.stack(new_images,0) if new_images_context is not [] else None
    def check_len(self,prev_img_dynamic_2s):
        missing_frames = self.fps - prev_img_dynamic_2s.shape[0]
        if missing_frames > 0:
            repeated_first_frame = prev_img_dynamic_2s[0:1].expand(missing_frames, -1, -1, -1)
            padded_frames = torch.cat([repeated_first_frame, prev_img_dynamic_2s], dim=0)
        else:
            padded_frames = prev_img_dynamic_2s
        return padded_frames
    def _load_image(self, root, filesplit, imgname):
        # Implement your image loading logic here
        # For example, using PIL:
        for split in filesplit:
            img_path = os.path.join(root, split, imgname)
            if os.path.exists(img_path):
                return Image.open(img_path)
        raise FileNotFoundError
    def EgoStatusFeatureBuilder(self, agent_input: AgentInput) -> torch.Tensor:
        """Inherited, see superclass."""
        ego_status = agent_input.ego_statuses[-1]#
        velocity = torch.tensor(ego_status.ego_velocity)
        acceleration = torch.tensor(ego_status.ego_acceleration)
        driving_command = torch.tensor(ego_status.driving_command)
        ego_status_feature = torch.cat([velocity, acceleration, driving_command], dim=-1)
        # return {"ego_status": ego_status_feature}
        return ego_status_feature
    def get_obs_from_agent_input(self, agent_input: AgentInput) -> torch.Tensor:
        resized_image = []
        for idx in [-3, -1]:
            cameras = agent_input.cameras[idx]  #
            # Crop to ensure 4:1 aspect ratio
            l0 = cameras.cam_l0.image[:, 104:-104] #[28:-28, 416:-416] #x=480x416/1920=104, y=270x28/1080=7
            f0 = cameras.cam_f0.image #[28:-28]
            r0 = cameras.cam_r0.image[:, 104:-104] #[28:-28, 416:-416]
            # stitch l0, f0, r0 images
            stitched_image = np.concatenate([l0, f0, r0], axis=1)
            # resized_image = cv2.resize(stitched_image, (1024, 256))
            tensor_image = transforms.ToTensor()(stitched_image)
            tensor_image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(tensor_image)
            resized_image.append(transforms.Resize((self.Con_resolution_h, 576), interpolation=transforms.InterpolationMode.BICUBIC)(tensor_image))
            # resized_image = transforms.Resize((self.Con_resolution_h, self.Con_resolution_w+), interpolation=transforms.InterpolationMode.BICUBIC)(tensor_image)
        return torch.stack(resized_image, dim=0)
    def __len__(self):
        return len(self.metric_cache_loader.tokens)

    def __getitem__(self, idx):
        token = self.metric_cache_loader.tokens[idx]
        # self.cache = metric_cache
        #input
        agent_input = self.scene_loader.get_agent_input_from_token(token)
        scene = self.scene_loader.get_scene_from_token(token)
        scene_log = scene.scene_metadata.log_name
        future_trajectory = self.compute_targets(scene)#gt trajectory，(8,3)

        prev_img, next_img = self._get_images(token)
        images_prev_miss, images_next_miss = self.prev_frames-len(prev_img), self.next_frames-len(next_img)
        assert images_prev_miss>=0 and images_next_miss>=0

        next_img = [prev_img[-2], prev_img[-1], *next_img]#adding current observation
        prev_img_context_1s, prev_img_dynamic_1s = self.data_augmentation(prev_img[:self.fps])
        prev_img_context_2s, prev_img_dynamic_2s = self.data_augmentation(prev_img[self.fps:])
        next_img_context, next_img_dynamic = self.data_augmentation(next_img)
        #check past frame
        prev_img_dynamic_2s = self.check_len(prev_img_dynamic_2s)
        next_img_dynamic = next_img_dynamic[1:]#Todo skip one history frame ,using only current observation as init for future frame generation
        # ego_status = agent_input.ego_statuses[-1] #need further process
        ego_status = self.EgoStatusFeatureBuilder(agent_input)# FOLLOW ego_mlp agent
        # model_obs_140o = self.get_obs_from_agent_input(agent_input)

        return self.AgentInput(token,
                               prev_img_context_1s,
                               prev_img_dynamic_1s,
                               prev_img_context_2s,
                               prev_img_dynamic_2s,
                               next_img_context,
                               next_img_dynamic,
                               scene_log,
                               ego_status,
                               future_trajectory)

class Dataset_mmu(Dataset):
    def __init__(self, config, split='train'):
        root_of_cc12m = config.dataset.mmu_data.cc12m
        # self.datalist = load_dataset(path=root_of_cc12m, split=split)
        print(root_of_cc12m)
        self.datalist = load_from_disk(root_of_cc12m)
        self.resolution = config.dataset.preprocessing.resolution
        self.center_crop = config.dataset.preprocessing.center_crop
        self.mode = config.dataset.und_type

        self.fixed_resolutions = [
            (256, 256),
            (320, 240),
            (240, 320),
            (368, 256),
            (256, 368),
        ]

    def __len__(self):
        return len(self.datalist)

    def collate_fn(self):
        return DataCollatorFormmu

    def image_transform(self, image, resolution=None, normalize=True):
        if resolution is None:
            resolution = (256, 256)

        image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
        image = transforms.ToTensor()(image)

        if normalize:
            image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)

        return image

    def select_resolution(self, original_width, original_height):
        """
        """
        aspect_ratio = original_width / original_height
        closest_resolution = min(self.fixed_resolutions, key=lambda res: abs((res[0] / res[1]) - aspect_ratio))
        return closest_resolution

    def __getitem__(self, i):
        sample = {}

        original_height, original_width = self.datalist[i]['image'].size
        resolution = self.select_resolution(original_width, original_height)

        sample['image'] = self.image_transform(self.datalist[i]['image'], resolution=resolution)
        sample['conversations'] = self.datalist[i]['conversations']
        sample['id'] = self.datalist[i]['id']
        sample['mode'] = self.mode

        return sample

class Dataset_lm(Dataset):
    def __init__(self, config, split='train'):
        root_of_fineweb = config.dataset.lm_data.fineweb
        self.datalist = load_dataset(path=root_of_fineweb, split=split)
        self.mode = config.dataset.lm_type
    def __len__(self):
        return len(self.datalist)
    def collate_fn(self):
        return DataCollatorForlm

    def __getitem__(self, i):
        samples = {}
        current_data = self.datalist[i]
        samples['text'] = current_data['text']
        samples['mode'] = self.mode
        samples['id'] = current_data['id']
        samples['language_score'] = current_data['language_score']
        return samples
class DatasetVQGAN(Dataset):
    def __init__(self,
                 config,
                 mode='VQGAN',
                 split='train',
                 augmentation_args=None):
        super(DatasetVQGAN, self).__init__()
        self.dataset_list = [(DataConfig[k][0],v) for k, v in config.dataset.vqgan_data.items()]
        self.split = split
        self.mode = mode
        self.json_root = config.dataset.json_root
        print(f'{split}_dataset use：', self.dataset_list)
        self.video_path = {i[1]: os.path.join(config.dataset.video_root[i[1]], i[0]['video_root']) for i in self.dataset_list}
        self.resolution_h, self.resolution_w = config.dataset.params.resolution_h, config.dataset.params.resolution_w
        self.Con_resolution_h, self.Con_resolution_w = config.dataset.params.Con_resolution_h, config.dataset.params.Con_resolution_w
        self.segment_horizon = config.dataset.params.segment_horizon
        if mode == 'VQGAN':
            self.segment_length = config.dataset.params.segment_length
            self.context_length = config.dataset.params.context_length
        self.vq_processor = image_transform

        # list_data_dict = []
        list_data_dict = []
        for i in self.dataset_list:
            list_data_dict = self.get_json_file(i, list_data_dict)
        self.list_data_dict = list_data_dict
        if self.split == 'train':
            self.list_data_dict = [x for i, x in enumerate(self.list_data_dict) if i % 100 != 0]
        else:
            self.list_data_dict = [x for i, x in enumerate(self.list_data_dict) if i % 100 == 0]
        if augmentation_args:
            self.aug = True
            self.random_resized_crop_scale = augmentation_args['random_resized_crop_scale']
            self.random_resized_crop_ratio = augmentation_args['random_resized_crop_ratio']
            self.brightness = augmentation_args['brightness']
            self.contrast = augmentation_args['contrast']
            self.saturation = augmentation_args['saturation']
            self.hue = augmentation_args['hue']

        else:
            self.aug = False
        # random.shuffle(self.list_data_dict)
        self.load_action = False
    def get_json_file(self, json_path, json_data):
        cur_json_data = []

        if 'opendvmini' in json_path[1].split('/'):
            if json_path[0]['Anno'] == '':
                if json_path['Video'] == '':
                    raise FileNotFoundError('Please provide Anno and video path')
                else:
                    all_files = []
                    for dirpath, dirnames, filenames in os.walk(self.video_path['opendvmini']):
                        for filename in filenames:
                            all_files.extend({'vid_path': os.path.join(dirpath, filename)})
                    return all_files
            else:
                with open(os.path.join(self.json_root, json_path[0]['Anno']), 'r', encoding='utf-8') as f:
                    json_data.extend(json.load(f))
            return json_data
        elif "opendv" in json_path[1].split('/'):

            if json_path[0]['Anno'] == '':
                if json_path['Video'] == '':
                    raise FileNotFoundError('Please provide Anno and video path')
                else:
                    all_files = []
                    for dirpath, dirnames, filenames in os.walk(self.video_path['opendv']):
                        for filename in filenames:
                            all_files.extend({'vid_path': os.path.join(dirpath, filename)})
                    return all_files
            else:
                with open(os.path.join(self.json_root, json_path[0]['Anno']), 'r', encoding='utf-8') as f:
                    json_data.extend(json.load(f))
            return json_data
    def collate_fn(self):
        return DataCollatorForvideo

    def data_augmentation(self, images):


        if self.aug:
            i, j, h, w = self.get_crop_params(images[0], self.random_resized_crop_scale)#, self.random_resized_crop_ratio)
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_jittor_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        con_len_count = 0
        new_images_context = []
        new_images = []
        tensor = transforms.ToTensor()
        for image in images:
            if self.aug:
                image = F.resized_crop(image, i, j, h, w, [self.resolution_h, self.resolution_w])
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        image = F.adjust_brightness(image, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        image = F.adjust_contrast(image, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        image = F.adjust_saturation(image, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        image = F.adjust_hue(image, hue_factor)
            else:
                if con_len_count < self.context_length: #context image process
                    image = transforms.Resize((self.Con_resolution_h, self.Con_resolution_w), interpolation=transforms.InterpolationMode.BICUBIC)(image)
                    image = transforms.CenterCrop((self.Con_resolution_h, self.Con_resolution_w))(image)
                    image = tensor(image) #/255
                    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
                    new_images_context.append(image)
                else:
                    image = transforms.Resize((self.resolution_h, self.resolution_w), interpolation=transforms.InterpolationMode.BICUBIC)(image)
                    image = transforms.CenterCrop((self.resolution_h, self.resolution_w))(image)
                    image = tensor(image) #/255
                    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
                    new_images.append(image)
            con_len_count += 1
        return torch.stack(new_images_context,0), torch.stack(new_images,0)
    def get_jittor_params(self,
            brightness: Optional[List[float]],
            contrast: Optional[List[float]],
            saturation: Optional[List[float]],
            hue: Optional[List[float]],
    ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def get_crop_params(self, img: Tensor, scale: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop with fixed aspect ratio.

        Args:
            img (Tensor): Input image (C, H, W).
            scale (list): Range of scale of the origin size cropped.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop, keeping the original aspect ratio.
        """
        _, height, width = F.get_dimensions(img)
        area = height * width

        original_ratio = width / height

        for _ in range(3):

            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()


            h = int(round(math.sqrt(target_area / original_ratio)))
            w = int(round(h * original_ratio))


            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w
        if width > height:
            w = int(round(height * original_ratio))
            h = height
        else:
            h = int(round(width / original_ratio))
            w = width

        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
    def get_image_path(self, sample_dict, current_index, dataset_name):
        if dataset_name == 'opendv' or dataset_name == 'opendvmini':
            first_frame = sample_dict["first_frame"]
            idx_str, ext_str = first_frame.split(".")
            format_length = len(idx_str)
            file_name = str(int(idx_str) + current_index).zfill(format_length) + "." + ext_str
            return os.path.join(self.video_path[dataset_name], sample_dict["folder"], file_name)
        else:
            raise NotImplementedError
    def get_segment(self, num_frames, start_index, end_index, select_num, short_flag=False, mode=None):
        start_index = start_index
        end_index = end_index
        if start_index >= num_frames or end_index >= num_frames:
            print(f'start_index: {start_index}, start_index: {end_index},total_frames: {num_frames}')
            raise ValueError("start_index must be less than num_frames")
        indices = np.arange(start_index, end_index)
        if mode == 'dynamic':
            weights = indices - start_index + 1
        else:
            weights = np.ones_like(indices)

        if short_flag:
            selected_indices = np.random.choice(indices, size=select_num, replace=True, p=weights/weights.sum())
        else:
            selected_indices = np.random.choice(indices, size=select_num, replace=False, p=weights/weights.sum())

        return selected_indices.tolist()
    def Opendv(self, sources, num_frames, context_num, select_num):
        samples = {}
        start_idx = np.random.choice(np.arange(0, num_frames - select_num * 2), replace=False)
        actions = None
        ids_context = sorted(
            self.get_segment(num_frames, start_idx, num_frames - select_num * 2 + 1, context_num, mode='context'))  # time step
        ids_dynmic = sorted(
            self.get_segment(num_frames, ids_context[-1] + 1, num_frames - 1, select_num - context_num, mode='dynamic'))  # time step
        ids = ids_context + ids_dynmic
        if 'set' in sources:#mini
            images = []
            for i in range(select_num):
                with Image.open(self.get_image_path(sources, ids[i], 'opendvmini')) as img:
                    images.append(img.convert('RGB').copy())
        else:
            images = []
            for i in range(select_num):
                try:
                    with Image.open(self.get_image_path(sources, ids[i], 'opendv')) as img:
                        images.append(img.convert('RGB').copy())
                except (OSError, UnidentifiedImageError) as e:
                    print(f"Warning: Skipping corrupted or invalid image at {self.get_image_path(sources, ids[i], 'opendv')} due to error: {e}")
                    flag_img = 1
                    j = 0
                    while flag_img and j <= 10:
                        j += 1
                        try:
                            with Image.open(self.get_image_path(sources, ids[i] + j, 'opendv')) as img:
                                images.append(img.convert('RGB').copy())
                                flag_img = False
                        except (OSError, UnidentifiedImageError) as e:
                            print(
                                f"Warning: Skipping corrupted or invalid image at {self.get_image_path(sources, ids[i] + j, 'opendv')} due to error: {e}")
                    if j > 10:
                        print(
                            f"Warning: Unable to load image after {j} retries for {self.get_image_path(sources, ids[i], 'opendv')}")
        context_iamges, images = self.data_augmentation(images)
        samples['time_step'] = [i - ids[0] for i in ids]
        samples['images'] = images
        samples['context_images'] = context_iamges
        return samples, actions

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        sources = self.list_data_dict[idx]
        select_num = self.segment_length
        context_num = self.context_length
        if 'cmd' in sources.keys():#opendv
            num_frames = int(sources['last_frame'].split('.')[0]) - int(sources['first_frame'].split('.')[0])
            samples, actions = self.Opendv(sources, num_frames, context_num, select_num)
        if self.load_action:
            actions = torch.Tensor(np.array(actions))
            samples['actions'] = actions
            return samples
        else:
            return samples

@dataclass
class DataCollatorForVQGAN(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        time_step = [instance["time_step"] for instance in instances if "time_step" in instance]
        batch = dict(
            time_step=time_step,
        )
        for i in ['images', 'context_images']:
            if i in instances[0]:
                state_action = [instance[i] for instance in instances]

                new_images = []
                for image in state_action:
                    if isinstance(image, list):
                        for i in image:
                            new_images.append(i)
                    else:
                        new_images.append(image)
                images = new_images

                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch[i] = torch.stack(images)
                else:
                    batch[i] = images

        return batch
@dataclass
class DataCollatorForSupervisedNuScenes(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_desc_a = [instance["desc_a"] for instance in instances if "desc_a" in instance]
        input_action_a = [instance["action_a"] for instance in instances if "action_a" in instance]
        input_token = [instance["token"] for instance in instances if "token" in instance]
        idx = [instance["idx"] for instance in instances if "idx" in instance]
        scene_name = [instance["scene_name"] for instance in instances if "scene_name" in instance]
        batch = dict(
            input_desc_a = input_desc_a,
            input_action_a = input_action_a,
            input_token = input_token,
            idx = idx,
            scene_name = scene_name,
        )

        for i in ['prev_img_context', 'prev_img_dynamic', 'next_img_context', 'next_img_dynamic','planning_a', 'real_len','ego_status','H_cmd','plan_mask']:
            if i in instances[0]:
                state = [instance[i] for instance in instances]

                new_images = []
                for image in state:
                    if isinstance(image, list):
                        for i in image:
                            new_images.append(i)
                    else:
                        new_images.append(image)
                images = new_images

                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch[i] = torch.stack(images)
                else:
                    batch[i] = images

        return batch

class DataCollatorForNavsim(object):
    """Collate examples for supervised fine-tuning."""

    # tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        token = [instance["token"] for instance in instances]
        scene_log = [instance["scene_log"] for instance in instances]
        batch = dict(
            token=token,
            scene_log=scene_log
        )
        for i in ['prev_img_context_1s', 'prev_img_dynamic_1s',
                  'prev_img_context_2s', 'prev_img_dynamic_2s',
                  'next_img_context', 'next_img_dynamic',
                  'ego_status', 'future_trajectory']:
            if i in instances[0]:
                list_insts = [instance[i] for instance in instances]

                Batchwise_inst = []
                for inst in list_insts:
                    if type(inst) is list:
                        for i in inst:
                            Batchwise_inst.append(i)
                    else:
                        Batchwise_inst.append(inst)
                # insts = Batchwise_inst
                if all(x is not None and x.shape == Batchwise_inst[0].shape for x in Batchwise_inst):
                    batch[i] = torch.stack(Batchwise_inst)
                else:
                    batch[i] = Batchwise_inst
        return batch

def DataCollatorForvideo(instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    time_step = [instance["time_step"] for instance in instances if "time_step" in instance]
    type = [instance["type"] for instance in instances if "type" in instance]
    mode = [instance["mode"] for instance in instances if "mode" in instance]
    caption = [instance["caption"] for instance in instances if "caption" in instance]
    actions = [instance["actions"] for instance in instances if "actions" in instance]
    batch = dict(time_step=time_step, type=type, mode=mode, caption=caption, actions=actions)
    for i in ['images', 'context_images']:
        if i in instances[0]:
            state_action = [instance[i] for instance in instances]

            new_images = []
            for image in state_action:
                if isinstance(image, list):
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images

            if all(x is not None and x.shape == images[0].shape for x in images):
                batch[i] = torch.stack(images)
            else:
                batch[i] = images # (B,T,C,H,W)

    return batch


def DataCollatorFormmu(instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    text = [instance["conversations"] for instance in instances if "conversations" in instance]
    id = [instance["id"] for instance in instances if "id" in instance]
    mode = [instance["mode"] for instance in instances if "mode" in instance]
    batch = dict(conversation=text, id=id, mode=mode)

    max_height = max(instance['image'].shape[1] for instance in instances)
    max_width = max(instance['image'].shape[2] for instance in instances)

    padded_images = []
    for sample in instances:
        image = sample['image']
        _, h, w = image.shape

        pad_top = (max_height - h) // 2
        pad_bottom = max_height - h - pad_top
        pad_left = (max_width - w) // 2
        pad_right = max_width - w - pad_left

        padded_image = torch.nn.functional.pad(
            image,
            (pad_left, pad_right, pad_top, pad_bottom),
            value=0
        )
        padded_images.append(padded_image)

    batch['image'] = torch.stack(padded_images)

    return batch


def DataCollatorForlm(instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    text = [instance["text"] for instance in instances if "text" in instance]
    id = [instance["id"] for instance in instances if "id" in instance]
    mode = [instance["mode"] for instance in instances if "mode" in instance]
    language_score = [instance["language_score"] for instance in instances if "language_score" in instance]
    batch = dict(text=text, id=id, mode=mode, language_score=language_score)
    return batch

def find_damaged_images(image_dir, output_file):
    damaged_images = []  #

    for root, dirs, files in tqdm(os.walk(image_dir)):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                img_path = os.path.join(root, file)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except (UnidentifiedImageError, IOError):
                    print(f"Damaged image found: {img_path}")
                    damaged_images.append(img_path)
    if output_file:
        with open(output_file, 'w') as f:
            for img in damaged_images:
                f.write(f"{img}\n")

    return damaged_images


if '__main__' == __name__:
    pass