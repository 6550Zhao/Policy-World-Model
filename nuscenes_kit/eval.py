import json
import argparse
import os
import math
import pickle
import torch
import numpy as np
from vid_dataset.gen_data import QA
from planning_utils import PlanningMetric
from metrics import calc_l2, eval_qa
from tqdm import tqdm
def main(args):
    anno_path = args.anno_path
    nusc_path = args.nusc_path
    save_path = args.save_path
    model_path = args.model_path
    data_path = args.data_path
    task = args.task

    with open(data_path, 'r') as f:
        data = json.load(f) # [[conv[qas]]]

    div = len(data) // args.split

    def qa_from_dict(d):
        qa = QA()
        qa.from_dict(d)
        return qa
    logs = list()
    # mask_tokens, mask_ids = encode_mask(processor)
    mask_ids =1234
    logs = [{'plan':np.random.randn(6,2),'gt':np.random.randn(6,2),'id': 0},{'plan':np.random.randn(6,2),'gt':np.random.randn(6,2),'id': 1},{'plan':np.random.randn(6,2),'gt':np.random.randn(6,2),'id': 2}]
    # metrics for plan and qa
    metric_dict = dict()
    if task == 'plan':
        planning_metric = PlanningMetric(nusc_path)
        with open(anno_path, 'rb') as f:
            annos = pickle.load(f)
        future_seconds = 3
        l2, cnt = np.zeros(2*future_seconds), 0
        # coll
        colls = [0., 0., 0.]
        
        for log in logs:#per batch
            
            if 'plan' in log:#log is dict
                l2 += np.array(calc_l2(log['plan'], log['gt']))
                plan = torch.tensor(log['plan']).unsqueeze(0)#(B=1,T,2)
                gt_infos = annos['infos'][log['id']]
                gt_agent_boxes = np.concatenate([gt_infos['gt_boxes'], gt_infos['gt_velocity']], -1)
                gt_agent_feats = np.concatenate([gt_infos['gt_fut_traj'][:, :6].reshape(-1, 12), gt_infos['gt_fut_traj_mask'][:, :6], gt_infos['gt_fut_yaw'][:, :6], gt_infos['gt_fut_idx']], -1)
                bev_seg = planning_metric.get_birds_eye_view_label(gt_agent_boxes, gt_agent_feats, add_rec=True)
                gt_traj = gt_infos['gt_planning']
                gt_traj = torch.from_numpy(gt_traj[..., :2])
                seg = torch.from_numpy(bev_seg[1:]).unsqueeze(0)
                for jj in range(future_seconds):#1s,2s,3s coll
                    cur_time = (jj+1)*2
                    _, coll = planning_metric.evaluate_coll(plan[:,:cur_time,:2], gt_traj[:,:cur_time,:], seg)
                    coll = coll.mean().item()
                    colls[jj] += coll

                cnt += 1
                
        for i in range(future_seconds):
            cur_time = (i+1)*2
            metric_dict[f'l2_{i+1}s'] = l2[:cur_time].sum().item() / cur_time / cnt
            metric_dict[f'coll_{i+1}s'] = colls[i] / cnt
        metric_dict['samples'] = cnt

    elif task == 'qa':
        predictions, references = [], []
        # batch_size = 1024

        for log in logs:
            if 'task' in log and log['task'] == 'qa':
                predictions.append(log['generate'])
                references.append(log['gt'])
                # add metrics
        qa_dict = eval_qa(predictions, references, 'meteor')
        for k, v in qa_dict.items():
            metric_dict[k] = v

    else:
        raise NotImplementedError()
    
    print(metric_dict)
    # TODO: support more evaluate metrics

    with open(os.path.join(save_path, f'result-{args.id}.json'), 'w') as f:
        json.dump(metric_dict, f, indent=4)

if __name__ == "__main__":
    pass
