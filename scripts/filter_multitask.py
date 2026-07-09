#!/usr/bin/env python3
"""
Filter merged_multitask dataset by commander_state.

Rules:
  - Non-HIL tasks: keep all frames (100% teleop)
  - HIL tasks, pure teleop episode: keep all
  - HIL tasks, pure inference episode: keep all (successful autonomous rollout)
  - HIL tasks, mixed episode: keep only teleop frames (discard inference that went wrong)
  - Always drop: align, pre_teleop, restore
  - Keep original episode_index and frame_index so video lookups stay correct.
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict

import pandas as pd
import numpy as np
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


MIN_FRAMES_PER_EPISODE = 50
HIL_ONLY_STATES = {"teleop"}
BAD_STATES = {"align", "pre_teleop", "restore"}


def load_episode_tasks(episodes_path: Path) -> Dict[int, str]:
    mapping = {}
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)
            mapping[ep["episode_index"]] = ep["tasks"]
    return mapping


def is_hil_task(task_name: str) -> bool:
    return "-hil" in task_name


def resolve_parquet_path(src_root: Path, src_info: dict, ep_idx: int) -> Path:
    chunksize = int(src_info.get("chunks_size", 1000))
    chunk_idx = ep_idx // chunksize
    rel = src_info["data_path"].format(episode_chunk=chunk_idx, episode_index=ep_idx)
    path = src_root / rel
    if path.is_file():
        return path
    basename = f"episode_{ep_idx:06d}.parquet"
    candidates = list(src_root.rglob(basename))
    return candidates[0] if candidates else path


def resolve_video_path(src_root: Path, src_info: dict, ep_idx: int, video_key: str) -> Path | None:
    chunksize = int(src_info.get("chunks_size", 1000))
    chunk_idx = ep_idx // chunksize
    try:
        rel = src_info["video_path"].format(
            episode_chunk=chunk_idx, video_key=video_key, episode_index=ep_idx
        )
        path = src_root / rel
        if path.is_file():
            return path
    except Exception:
        pass
    basename = f"episode_{ep_idx:06d}.mp4"
    candidates = [p for p in src_root.rglob(basename) if video_key in str(p)]
    return candidates[0] if candidates else None


def main():
    src_path = Path("/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/merged_multitask")
    tgt_path = Path("/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge/filtered_multitask")
    repo_id = "filtered_multitask"

    if tgt_path.exists() and any(tgt_path.iterdir()):
        raise RuntimeError(f"Target {tgt_path} exists and is not empty. Remove it first.")

    with open(src_path / "meta" / "info.json") as f:
        src_info = json.load(f)
    episode_tasks = load_episode_tasks(src_path / "meta" / "episodes.jsonl")
    features = src_info.get("features", {})

    ds_target = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(src_info.get("fps", 60)),
        root=str(tgt_path),
        robot_type=src_info.get("robot_type", "yam"),
        features=features,
    )
    meta_target = ds_target.meta
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]

    total_in_frames = 0
    total_out_frames = 0
    total_in_eps = 0
    total_out_eps = 0
    dropped_frames = Counter()
    kept_frames = Counter()
    task_eps = Counter()

    items = sorted(episode_tasks.items(), key=lambda kv: kv[0])
    for src_ep_idx, task_name in tqdm(items, desc="Processing"):
        total_in_eps += 1

        src_parquet = resolve_parquet_path(src_path, src_info, src_ep_idx)
        df = pd.read_parquet(src_parquet)
        total_in_frames += len(df)

        col = "observation.commander_state"

        if is_hil_task(task_name):
            commander = df[col].astype(str)
            ep_states = set(commander.unique()) - BAD_STATES

            is_pure_teleop = ep_states == {"teleop"}
            is_pure_inference = ep_states == {"inference"}

            if is_pure_teleop or is_pure_inference:
                # Pure episode: keep all, only drop align/pre_teleop/restore
                keep_mask = ~commander.isin(BAD_STATES)
            else:
                # Mixed episode: inference went wrong, keep only teleop
                keep_mask = commander.isin(HIL_ONLY_STATES)

            for s in commander.unique():
                cnt = int((commander == s).sum())
                actual_kept = int((keep_mask & (commander == s)).sum())
                if actual_kept > 0:
                    kept_frames[s] += actual_kept
                if cnt - actual_kept > 0:
                    dropped_frames[s] += cnt - actual_kept

            df = df[keep_mask].copy()
            # Regenerate continuous timestamps so action chunk lookups don't span gaps.
            fps = int(src_info.get("fps", 60))
            df["timestamp"] = [i / fps for i in range(len(df))]

        else:
            # Non-HIL: already 100% teleop, keep all
            kept_frames["teleop"] += len(df)

        if len(df) < MIN_FRAMES_PER_EPISODE:
            continue

        # Register task if needed
        if meta_target.get_task_index(task_name) is None:
            meta_target.add_task(task_name)

        # Keep original episode_index (src_ep_idx) so video lookups stay correct
        new_ep_idx = src_ep_idx

        # Write filtered parquet to target, using original episode_index for path
        tgt_chunk_idx = meta_target.get_episode_chunk(new_ep_idx)
        tgt_parquet_rel = meta_target.data_path.format(
            episode_chunk=tgt_chunk_idx, episode_index=new_ep_idx
        )
        tgt_parquet_path = meta_target.root / tgt_parquet_rel
        tgt_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(tgt_parquet_path), index=False)

        # Symlink videos (original episode index → same path in target)
        for vk in video_keys:
            src_vid = resolve_video_path(src_path, src_info, src_ep_idx, vk)
            if src_vid is None:
                continue
            tgt_vid_rel = meta_target.video_path.format(
                episode_chunk=tgt_chunk_idx, video_key=vk, episode_index=new_ep_idx
            )
            tgt_vid_path = meta_target.root / tgt_vid_rel
            tgt_vid_path.parent.mkdir(parents=True, exist_ok=True)
            if not tgt_vid_path.exists():
                tgt_vid_path.symlink_to(src_vid)

        meta_target.save_episode(
            episode_index=new_ep_idx,
            episode_length=len(df),
            episode_tasks=[task_name],
            episode_stats={},
        )

        total_out_frames += len(df)
        total_out_eps += 1
        task_eps[task_name] += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"Input episodes:  {total_in_eps}")
    print(f"Output episodes: {total_out_eps}")
    print(f"Input frames:    {total_in_frames:,}")
    print(f"Output frames:   {total_out_frames:,} ({100*total_out_frames/total_in_frames:.1f}%)")
    print(f"\nFrames KEPT by state:")
    for s, c in kept_frames.most_common():
        print(f"  {s}: {c:,}")
    if dropped_frames:
        print(f"\nFrames DROPPED by state:")
        for s, c in dropped_frames.most_common():
            print(f"  {s}: {c:,}")
    print(f"\nOutput episodes by task:")
    for t, c in sorted(task_eps.items()):
        tag = " [HIL: teleop only]" if is_hil_task(t) else " [all teleop, no filtering needed]"
        print(f"  {t}: {c}{tag}")

    # Smoke test
    ds_check = LeRobotDataset(repo_id=repo_id, root=meta_target.root)
    print(f"\n[*] Target loaded OK: {ds_check.num_episodes} episodes, {ds_check.num_frames:,} frames")


if __name__ == "__main__":
    main()
