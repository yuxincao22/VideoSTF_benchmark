import argparse
import os

from models import register_adapters
from stressors.temporal import (AddOneFrame, AddTwoFrames, DeleteOneFrame, DeleteTwoFrames,
                                ReplaceOneFrame, ReplaceTwoFrames, ReverseVideo, RandomShuffle)
from utils.env import set_env_sane_defaults, pin_gpu
from utils.io import safe_makedirs, list_videos
from utils.registry import REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Batch video inference")

    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--max_frames_num", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--gpu", type=str, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Please describe this video in detail.")

    args = parser.parse_args()

    return args

def main():
    set_env_sane_defaults()
    cfg = parse_args()

    pin_gpu(cfg.gpu)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    cfg.output_folder = os.path.join(root_dir, "test_results", cfg.output_folder)
    safe_makedirs(cfg.output_folder)

    register_adapters(cfg.adapter)
    adapter_cls = REGISTRY.get(cfg.adapter)
    adapter = adapter_cls(cfg)

    print(f"Loading adapter: {cfg.adapter}")
    adapter.load()

    prompt = cfg.prompt

    transformations = [
        AddOneFrame(times=30),
        AddTwoFrames(times=30),
        DeleteOneFrame(),
        DeleteTwoFrames(times=30),
        ReplaceOneFrame(times=30),
        ReplaceTwoFrames(times=30),
        ReverseVideo(),
        RandomShuffle(times=30),
    ]

    videos = list_videos(cfg.input_folder)
    if not videos:
        print(f"No videos found in {cfg.input_folder}")
        return

    print(f"Found {len(videos)} videos")
    for i, video_path in enumerate(videos, start=1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        csv_path = os.path.join(cfg.output_folder, f"{video_name}.csv")
        print(f"[{i}/{len(videos)}] {os.path.basename(video_path)}")
        adapter.stress_test(video_path, video_name, csv_path, transformations, prompt)
    print("Done.")

if __name__ == "__main__":
    main()
