import argparse
import os

from utils.cal_metrics import cal, cal_batch
from utils.env import set_env_sane_defaults, pin_gpu
from utils.io import safe_makedirs, list_videos, write_json
from utils.registry import REGISTRY
from models import register_adapters

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

def process_one(adapter, video_path, video_name, prompt):
    output = adapter.generate(video_path, prompt)
    rr_value, ri_value, ie_value = cal(output)
    print(f"RR: {rr_value}, RI: {ri_value}, IE: {ie_value}")

    return {
        "video_path": video_path,
        "video_name": video_name,
        "adapter": adapter.name,
        "temperature": adapter.cfg.temperature,
        "max_frames_num": adapter.cfg.max_frames_num,
        "max_new_tokens": adapter.cfg.max_new_tokens,
        "prompt": prompt,
        "output": output,
        "rr_value": rr_value,
        "ri_value": ri_value,
        "ie_value": ie_value,
    }

def main():
    set_env_sane_defaults()
    cfg = parse_args()

    pin_gpu(cfg.gpu)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    cfg.output_folder = os.path.join(root_dir, "infer_results", cfg.output_folder)
    safe_makedirs(cfg.output_folder)

    register_adapters(cfg.adapter)
    adapter_cls = REGISTRY.get(cfg.adapter)
    adapter = adapter_cls(cfg)

    print(f"Loading adapter: {cfg.adapter}")
    adapter.load()

    prompt = cfg.prompt

    outputs = []

    videos = list_videos(cfg.input_folder)
    if not videos:
        print(f"No videos found in {cfg.input_folder}")
        return

    print(f"Found {len(videos)} videos")
    for i, video_path in enumerate(videos, start=1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(cfg.output_folder, f"{video_name}.json")
        err_path = os.path.join(cfg.output_folder, f"{video_name}.error.json")

        print(f"[{i}/{len(videos)}] {os.path.basename(video_path)}")
        try:
            res = process_one(adapter, video_path, video_name, prompt)
            outputs.append(res["output"])
            write_json(out_path, res)
            print(f"Saved: {out_path}")
        except Exception as e:
            write_json(err_path, {
                "video_path": video_path,
                "video_name": video_name,
                "adapter": cfg.adapter,
                "error": repr(e),
            })
            print(f"Failed: {video_path}")
            print(f"Error saved: {err_path}")
    rr_value_avg, ri_value_avg, ie_value_avg = cal_batch(outputs)
    print(f"RR_avg: {rr_value_avg}, RI_avg: {ri_value_avg}, IE_avg: {ie_value_avg}")
    print("Done.")

if __name__ == "__main__":
    main()
