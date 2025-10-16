from PIL import Image
import math
from decord import VideoReader, cpu
import numpy as np
import os
import torch

OPENDV_LOCAL = {
    "Anno": "mini_train_annos.json",
    "video_root": "OPendvmini",

}
OPENDV_MINI = {
    "Anno": "opendvmini/mini_opendv_add_conversation.json",
    "video_root": "/defaultShare/OPendvmini",
}

OPENDV_LOCAL_val = {
    "Anno": "mini_val_annos.json",
    "video_root": "OPendvmini",

}

OPENDV_FULL = {

    "Anno": 'dataset/opendv/opendv_caption.json',
    "video_root": "/OpenDV-YouTube",
}
NUSCENES_FRONT = {
    "Anno": "nuscenes/nuscenes_CAM_FRONT_data.json",
    "video_root": "/nuscenes",
}

NUSCENES_BACK= {
    "Anno": "nuscenes/nuscenes_CAM_BACK_data.json",
    "video_root": "/nuscenes",
}

NUSCENES_FRONT_LEFT= {
    "Anno": "nuscenes/nuscenes_CAM_FRONT_LEFT_data.json",
    "video_root": "/nuscenes",
}

NUSCENES_FRONT_RIGHT= {
    "Anno": "nuscenes/nuscenes_CAM_FRONT_RIGHT_data.json",
    "video_root": "/nuscenes",
}
NUSCENES_BACK_LEFT= {
    "Anno": "nuscenes/nuscenes_CAM_BACK_LEFT_data.json",
    "video_root": "/nuscenes",
}

NUSCENES_BACK_RIGHT= {
    "Anno": "nuscenes/nuscenes_CAM_BACK_RIGHT_data.json",
    "video_root": "/nuscenes",
}
def process_coco_image(image_ids, data_dict, list_data_dict, data_path):
    for image_id in image_ids:
        image_info = data_dict.loadImgs(image_id)[0]
        image_path = os.path.join(data_path, 'COCO/images/train2017', image_info['file_name'])
        caption_ids = data_dict.getAnnIds(imgIds=image_id)
        captions = data_dict.loadAnns(caption_ids)
        for caption in captions:
            list_data_dict.append({'image': image_path, 'caption': caption['caption']})
    return list_data_dict
def _get_rawvideo_dec(video_path, max_frames=64, image_resolution=224, video_framerate=8, s=None, e=None):
    # speed up video decode via decord.
    video_mask = np.zeros(max_frames, dtype=np.int64)
    max_video_length = 0

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(160 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = 6
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            start_sample = torch.randint(len(all_pos)-max_frames, (1, )).item()
            sample_pos = all_pos[start_sample:start_sample+max_frames]
        else:
            sample_pos = all_pos#采样位置

        patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
        slice_len = len(patch_images)
        return  patch_images, slice_len
        max_video_length = max_video_length if max_video_length > slice_len else slice_len
        if slice_len < 1:
            pass
        else:
            while len(patch_images) < max_frames:
                patch_images.append(torch.zeros((3, image_resolution, image_resolution)))
    else:
        print("video path: {} error.".format(video_path))

    video_mask[:max_video_length] = [1] * max_video_length

    return patch_images, video_mask