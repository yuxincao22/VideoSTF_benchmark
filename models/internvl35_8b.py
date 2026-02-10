import json
import os

import torch
try:
    from transformers import AutoModel, AutoTokenizer
except:
    print("Failed to import!")

from metrics.rr import max_ngram_repeats
from models.base import BaseAdapter
from utils.cal_metrics import cal_batch
from utils.utils_csv import append_attack_csv, load_existing_transformations, append_csv
from utils.video import load_video_raw_internvl3_5, preprocess_video_internvl3_5


class InternVL358BAdapter(BaseAdapter):
    name = "internvl35_8b"

    def load(self):
        model_path = "OpenGVLab/InternVL3_5-8B"
        device_map = "auto"

        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map,
            cache_dir=self.cfg.cache_dir).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False,
                                                  cache_dir=self.cfg.cache_dir)
        self.tokenizer = tokenizer
        self.model = model
        self.print_model_name = model_path.split("/")[-1]
        self.temperature = self.cfg.temperature
        self.max_frames_num = self.cfg.max_frames_num
        self.output_folder = self.cfg.output_folder
        self.max_new_tokens = self.cfg.max_new_tokens

    def inference(self, video_ori, prompt, rr_flag=False):
        pixel_values, num_patches_list = preprocess_video_internvl3_5(video_ori)
        video = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + prompt
        generation_config = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
        )
        text_outputs, history = self.model.chat(self.tokenizer, video, question, generation_config,
                                                num_patches_list=num_patches_list, history=None, return_history=True)
        print(text_outputs)
        if rr_flag:
            max_count, max_ngrams = max_ngram_repeats(text_outputs, n=self.n, by_word=True)
            return text_outputs, max_count, max_ngrams
        return text_outputs

    def generate(self, video_path, prompt):
        video_ori = load_video_raw_internvl3_5(video_path, num_segments=self.max_frames_num)
        text_outputs = self.inference(video_ori, prompt)
        return text_outputs

    def run_single_transformation(self, video_ori, video_path, video_name, transformation, prompt):
        outputs = []
        transformed_list = transformation.apply(video_ori)

        for transformed_video, setting in transformed_list:
            print("transformation: ", transformation.name)
            print("transformation setting: ", setting)
            text_outputs, max_count, max_ngrams = self.inference(transformed_video, prompt, rr_flag=True)
            outputs.append(text_outputs)

            repeat_flag = True if max_count > self.threshold else False
            max_ngrams = [] if max_count == 1 else max_ngrams

            output_data = {
                "video_path": video_path,
                "video_name": video_name,
                "max_frames_num": self.max_frames_num,
                "output": text_outputs,
                "max_count": max_count,
                "max_ngrams": max_ngrams,
                "repeat_flag": repeat_flag,
                "transformation_setting": setting,
                "transformation_name": transformation.name,
            }

            json_path = os.path.join(
                self.output_folder,
                f"{video_name}_{transformation.name}_{str(setting)}.json"
            )

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)

        return outputs

    def stress_test(self, video_path, video_name, csv_path, transformations, prompt):
        video_ori = load_video_raw_internvl3_5(video_path, num_segments=self.max_frames_num)

        tested_transformations = load_existing_transformations(
            csv_path,
            self.print_model_name,
            self.max_frames_num
        )

        for transformation in transformations:
            if transformation.name in tested_transformations:
                print(f"Skip {transformation.name}, already tested")
                continue

            print(f"Running {transformation.name}")
            outputs = self.run_single_transformation(video_ori, video_path, video_name, transformation, prompt)
            total_num = len(outputs)
            rr_value_avg, ri_value_avg, ie_value_avg = cal_batch(outputs)
            print(f"RR_avg: {rr_value_avg}, RI_avg: {ri_value_avg}, IE_avg: {ie_value_avg}")

            append_csv(
                csv_path,
                {
                    "video_name": video_name,
                    "model": self.print_model_name,
                    "frames": self.max_frames_num,
                    "transformation": transformation.name,
                    "total_num": total_num,
                    "RR_avg": rr_value_avg,
                    "RI_avg": ri_value_avg,
                    "IE_avg": ie_value_avg,
                }
            )

    def attack(self, video_path, video_name, csv_path, transformations, prompt, finished_attacks):
        video_ori = load_video_raw_internvl3_5(video_path, num_segments=self.max_frames_num)
        text_outputs, max_count, max_ngrams = self.inference(video_ori, prompt, rr_flag=True)
        repeat_flag = True if max_count > self.threshold else False
        if repeat_flag:
            print(f"Skip {video_name}: NOT normal output")
            return

        for transformation in transformations:
            key = (video_name, transformation.name)
            if key in finished_attacks:
                print("Skip", transformation.name, "already attacked")
                continue

            query = 0
            print("Trying transformation:", transformation.name)

            transformed_list = transformation.apply(video_ori)

            for transformed_video, setting in transformed_list:
                query += 1
                print("transformation setting: ", setting)
                text_outputs, max_count, max_ngrams = self.inference(transformed_video, prompt, rr_flag=True)
                repeat_flag = True if max_count > self.threshold else False
                max_ngrams = [] if max_count == 1 else max_ngrams

                output_data = {
                    "video_path": video_path,
                    "video_name": video_name,
                    "max_frames_num": self.max_frames_num,
                    "output": text_outputs,
                    "max_count": max_count,
                    "max_ngrams": max_ngrams,
                    "repeat_flag": repeat_flag,
                    "transformation_setting": setting,
                    "transformation_name": transformation.name,
                }

                json_path = os.path.join(self.output_folder, f"{video_name}_{transformation.name}_{str(setting)}.json")

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

                print(f"Saved result to {json_path}")

                if repeat_flag:
                    print("Attack successful with query: ", query)
                    break
            if not repeat_flag:
                print("Attack failed with query: ", query)
            append_attack_csv(
                csv_path,
                {
                    "video_name": video_name,
                    "transformation": transformation.name,
                    "success": int(repeat_flag),
                    "query": query,
                }
            )
            finished_attacks.add(key)
