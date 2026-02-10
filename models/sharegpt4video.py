import json
import os

from PIL import Image
import torch

from utils.env import activate_repo
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
activate_repo(os.path.join(_THIS_DIR, "ShareGPT4Video"))

try:
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model
except:
    print("Failed to import!")

from metrics.rr import max_ngram_repeats
from models.base import BaseAdapter
from utils.cal_metrics import cal_batch
from utils.utils_csv import append_attack_csv, load_existing_transformations, append_csv
from utils.video import create_frame_grid, resize_image_grid, load_video_frames_sg4v


class ShareGPT4VideoAdapter(BaseAdapter):
    name = "sharegpt4video"

    def load(self):
        model_path = "Lin-Chen/sharegpt4video-8b"
        model_name = "sharegpt4video-8b"
        device_map = "auto"

        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map=device_map,
            cache_dir=self.cfg.cache_dir,
        )
        model.eval()

        self.conv_template = "llava_llama_3"
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.print_model_name = model_path.split("/")[-1]
        self.temperature = self.cfg.temperature
        self.max_frames_num = self.cfg.max_frames_num
        self.output_folder = self.cfg.output_folder
        self.max_new_tokens = self.cfg.max_new_tokens

    def inference(self, video_ori, prompt, top_p=0.9, num_beams=1, rr_flag=False):
        img_grid = create_frame_grid(video_ori, 50)
        img_grid = Image.fromarray(img_grid).convert("RGB")
        img_grid = resize_image_grid(img_grid)
        if not isinstance(img_grid, (list, tuple)):
            img_grid = [img_grid]
        image_size = img_grid[0].size
        video = process_images(img_grid, self.image_processor, self.model.config)[0]

        pre_query_prompt = "The provided image arranges keyframes from a video in a grid view, keyframes are separated with white bands. Answer concisely with overall content and context of the video, highlighting any significant events, characters, or objects that appear throughout the frames."
        question = DEFAULT_IMAGE_TOKEN + '\n' + pre_query_prompt + prompt

        conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device, non_blocking=True)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token is not None else self.tokenizer.eos_token_id

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video.to(dtype=torch.float16, device=self.model.device, non_blocking=True),
                image_sizes=[image_size],
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=pad_token_id,
                use_cache=True,
            )
            text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(text_outputs)
        if rr_flag:
            max_count, max_ngrams = max_ngram_repeats(text_outputs, n=self.n, by_word=True)
            return text_outputs, max_count, max_ngrams
        return text_outputs

    def generate(self, video_path, prompt):
        video_ori = load_video_frames_sg4v(video_path, num_segments=self.max_frames_num)
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
        video_ori = load_video_frames_sg4v(video_path, num_segments=self.max_frames_num)

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
        video_ori = load_video_frames_sg4v(video_path, num_segments=self.max_frames_num)
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
