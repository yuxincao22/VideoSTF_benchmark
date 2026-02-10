import copy
import json
import os

from utils.env import activate_repo
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
activate_repo(os.path.join(_THIS_DIR, "LLaVA-NeXT"))

try:
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
from utils.video import load_video


class LlavaNextVideo7BDPOAdapter(BaseAdapter):
    name = "llava_next_video_7b_dpo"

    def load(self):
        model_path = "lmms-lab/LLaVA-NeXT-Video-7B-DPO"
        model_name = "LLaVA-NeXT-Video-7B-DPO"
        device_map = "auto"

        overwrite_config = {
            "mm_spatial_pool_mode": "average",
            "mm_spatial_pool_stride": 2,
            "mm_newline_position": "no_token",
            "max_sequence_length": 4096 * 2,
            "tokenizer_model_max_length": 4096 * 2,
        }

        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map=device_map,
            overwrite_config=overwrite_config,
            cache_dir=self.cfg.cache_dir,
        )
        model.eval()

        self.conv_template = "vicuna_v1"
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.add_time_instruction = False
        self.print_model_name = model_path.split("/")[-1]
        self.temperature = self.cfg.temperature
        self.max_frames_num = self.cfg.max_frames_num
        self.output_folder = self.cfg.output_folder
        self.max_new_tokens = self.cfg.max_new_tokens

    def inference(self, video_ori, video_time, frame_time, prompt, rr_flag=False):
        video = self.image_processor.preprocess(video_ori, return_tensors="pt")["pixel_values"].cuda().half()
        video = [video]
        time_instruction = (f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly "
                            f"sampled from it. These frames are located at {frame_time}.Please answer the following "
                            f"questions related to this video.")
        if self.add_time_instruction:
            question = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\n" + prompt
        else:
            question = DEFAULT_IMAGE_TOKEN + f"\n" + prompt

        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).cuda()

        cont, image_features = self.model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=True if self.temperature > 0 else False,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        print(text_outputs)
        if rr_flag:
            max_count, max_ngrams = max_ngram_repeats(text_outputs, n=self.n, by_word=True)
            return text_outputs, max_count, max_ngrams
        return text_outputs

    def generate(self, video_path, prompt):
        video_ori, frame_time, video_time = load_video(video_path, self.max_frames_num, 1, force_sample=True)
        text_outputs = self.inference(video_ori, video_time, frame_time, prompt)
        return text_outputs

    def run_single_transformation(self, video_ori, video_path, video_name, frame_time, video_time,
                                  transformation, prompt):
        outputs = []
        transformed_list = transformation.apply(video_ori)

        for transformed_video, setting in transformed_list:
            print("transformation: ", transformation.name)
            print("transformation setting: ", setting)
            text_outputs, max_count, max_ngrams = self.inference(transformed_video, video_time, frame_time, prompt,
                                                                 rr_flag=True)
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
        video_ori, frame_time, video_time = load_video(video_path, self.max_frames_num, 1, force_sample=True)

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
            outputs = self.run_single_transformation(video_ori, video_path, video_name, frame_time,
                                                          video_time, transformation, prompt)
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
        video_ori, frame_time, video_time = load_video(video_path, self.max_frames_num, 1, force_sample=True)
        text_outputs, max_count, max_ngrams = self.inference(video_ori, video_time, frame_time, prompt, rr_flag=True)
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
                text_outputs, max_count, max_ngrams = self.inference(transformed_video, video_time, frame_time, prompt,
                                                                     rr_flag=True)
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
