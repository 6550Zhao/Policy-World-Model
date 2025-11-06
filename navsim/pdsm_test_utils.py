from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from nuplan.planning.script.builders.logging_builder import build_logger
# from vid_dataset.video_dataset_2xdownSampling import DatasetNavsim
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union
import os
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig,Scene,Trajectory,NAVSIM_INTERVAL_LENGTH
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from hydra.utils import instantiate
from pathlib import Path
import lzma
import pickle
import os
from functools import partial
import torch
import logging
from dataclasses import dataclass, field, asdict
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from datetime import datetime
from omegaconf import DictConfig
from PIL import Image
import uuid
import traceback
from nuplan.planning.utils.multithreading.worker_utils import worker_map
def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]], logger=None) -> List[Dict[str, Any]]:
    """
    Helper function to run PDMS evaluation in.针对一个线程而言
    :param args: input arguments
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    # logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")
    tokens = [a["token"] for a in args]
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    metric_cache_loader = MetricCacheLoader(args[0]["cache_path"])
    pdm_results: List[Dict[str, Any]] = []
    for idx,sample in enumerate(args):
        token = sample["token"]
        future_trajectory = sample["future_trajectory"]
        trajectory = Trajectory(future_trajectory, TrajectorySampling(
                num_poses=len(future_trajectory),
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),)
        # logger.info(
        #     f"Processing scenario {idx + 1} / {len(args)}"# in thread_id={thread_id}, node_id={node_id}"
        # )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            metric_cache_path = metric_cache_loader.metric_cache_paths[token].split('navtest_cache/')[-1]
            with lzma.open(os.path.join(sample['cache_path'], metric_cache_path), "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
            pdm_result = pdm_score(#只需要传给其预测的轨迹
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    return pdm_results
def PDSM_eval(config, samples_eval, logger):
    split = 'test'
    os.environ["NUPLAN_MAPS_ROOT"] = os.path.join(config.dataset.navsim_root, "maps")
    len_p = 0
    scoring_params = OmegaConf.load(config.dataset.scoring_path)
    # pdm_results: List[Dict[str, Any]] = []
    # score_rows: List[Tuple[Dict[str, Any], int, int]]
    simulator: PDMSimulator = instantiate(scoring_params.simulator)
    scorer: PDMScorer = instantiate(scoring_params.scorer)
    assert (simulator.proposal_sampling == scorer.proposal_sampling), "Simulator and scorer proposal sampling has to be identical"
    cache_name = config.dataset.scene_filter.train_cache if split == 'train' else config.dataset.scene_filter.test_cache
    cache_path = os.path.join(config.experiment.base_root, 'dataset/navsim/cache', cache_name)
    # self.scoring_params = OmegaConf.load(config.dataset.scoring_path)
    metric_cache_loader = MetricCacheLoader(Path(cache_path))
    # build_logger(config)
    worker = build_worker(config)
    logger.info(f"batch number of samples : {len(samples_eval)}")

    run_pdm_score_with_logger = partial(run_pdm_score, logger=logger)
    score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score_with_logger, samples_eval)


    return score_rows #average_row
def pdsm_score_process(config, score_rows, global_step,logger):
    logger.info(f"total number of samples : {len(score_rows)}")
    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(
        skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(os.path.join(config.experiment.output_dir))
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"step_{global_step}_{timestamp}.csv")

    logger.info(
        f"""
                    Finished running evaluation.
                        Number of successful scenarios: {num_sucessful_scenarios}.
                        Number of failed scenarios: {num_failed_scenarios}.
                        Final average score of valid results: {pdm_score_df['score'].mean()}.
                        Results are stored in: {save_path / f"step_{global_step}_{timestamp}.csv"}.
                    """
    )
    return average_row