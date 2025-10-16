import math
import pickle
import torch
import numpy as np
from nuscenes_kit.planning_utils import PlanningMetric


class Planning_Evaluator_mask(PlanningMetric):
    def __init__(self, nuscenes_data_path, trj_anno_path, future_seconds=3):
        super(Planning_Evaluator_mask, self).__init__(step=6)
        self.nuscenes_data_path = nuscenes_data_path
        self.trj_anno_path = trj_anno_path
        with open(self.trj_anno_path, 'rb') as f:
            self.annos = pickle.load(f)
        self.future_seconds = future_seconds
        # self.val_valid_num = np.array([[0.,0.]]*6) #[5569, 5419, 5269, 5119, 4969, 4819] - number of vaild samples for each timestep
        # self.colls = [0., 0., 0.]

    def calc_l2(self, plan, gt, mask=None):
        l2_ = [0.] * 6
        if mask is not None:
            for i, p in enumerate(plan):
                l2_[i] += math.sqrt(((p[0] - gt[i][0])**2)*mask[i][0] + ((p[1] - gt[i][1])**2)*mask[i][1])
        else:
            for i, p in enumerate(plan):
                l2_[i] += math.sqrt((p[0] - gt[i][0])**2 + (p[1] - gt[i][1])**2)
        return l2_
    def calc_ade(self, plan, gt):
        ADE = np.mean(np.sqrt(((plan[:, :, :2] - gt[:, :, :2]) ** 2).sum(axis=-1)))

        FDE = np.mean(np.sqrt(((plan[:, -1, :2] - gt[:, -1, :2]) ** 2).sum(axis=-1)))

        return ADE, FDE
    def eval(self, l2, cnt, colls, logs):
        analys = {}
        val_valid_num = np.array([[0., 0.]] * 6)
        for log in logs:  # per batch

            if 'plan' in log:
                cur_l2 = self.calc_l2(log['plan'], log['gt'], log['mask'])
                val_valid_num += np.array(log['mask'])
                analys[log['id'].item()] = cur_l2
                l2 += np.array(cur_l2)
                plan = torch.tensor(log['plan']).unsqueeze(0)  # (B=1,T,2)
                gt_infos = self.annos['infos'][log['id']]
                gt_agent_boxes = np.concatenate([gt_infos['gt_boxes'], gt_infos['gt_velocity']], -1)
                gt_agent_feats = np.concatenate([gt_infos['gt_fut_traj'][:, :6].reshape(-1, 12), gt_infos['gt_fut_traj_mask'][:, :6],
                     gt_infos['gt_fut_yaw'][:, :6], gt_infos['gt_fut_idx']], -1)
                bev_seg = self.get_birds_eye_view_label(gt_agent_boxes, gt_agent_feats, add_rec=True)
                gt_traj = gt_infos['gt_planning']
                gt_traj = torch.from_numpy(gt_traj[..., :2])
                seg = torch.from_numpy(bev_seg[1:]).unsqueeze(0)
                for jj in range(self.future_seconds):  # 1s,2s,3s coll
                    cur_time = (jj + 1) * 2
                    _, coll = self.evaluate_coll(plan[:, :cur_time, :2], gt_traj[:, :cur_time, :], seg)
                    coll = (coll*log['mask'][cur_time-1][0]).mean().item()
                    colls[jj] += coll

                cnt += 1

        metric_dict = dict()
        for i in range(self.future_seconds):
            cur_time = (i + 1) * 2
            metric_dict[f'l2_vad{i + 1}s'] = (l2[:cur_time]/val_valid_num[:cur_time,0]).sum().item()/cur_time#l2[:cur_time].sum().item() / cur_time / cnt
            metric_dict[f'l2_uniad{i + 1}s'] = l2[cur_time-1]/val_valid_num[cur_time-1,0]
            metric_dict[f'coll_{i + 1}s'] = colls[i] / val_valid_num[cur_time-1,0]#some samples have invalid GT
        metric_dict['samples'] = cnt

        return metric_dict, analys

