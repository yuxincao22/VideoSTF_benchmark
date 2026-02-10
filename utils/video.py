from decord import VideoReader, cpu
import numpy as np
from PIL import Image


## llava & qwen3vl & molmo2
def load_video(video_path, max_frames_num=32, fps=1, force_sample=False, return_idx=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    if return_idx:
        return spare_frames, frame_time, video_time, frame_idx
    return spare_frames, frame_time, video_time

## sharegpt4video
def create_frame_grid(img_array, interval_width=50):
    n, h, w, c = img_array.shape
    grid_size = int(np.ceil(np.sqrt(n)))

    horizontal_band = np.ones((h, interval_width, c),
                              dtype=img_array.dtype) * 255
    vertical_band = np.ones((interval_width, w + (grid_size - 1)
                             * (w + interval_width), c), dtype=img_array.dtype) * 255

    rows = []
    for i in range(grid_size):
        row_frames = []
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < n:
                frame = img_array[idx]
            else:
                frame = np.ones_like(img_array[0]) * 255
            if j > 0:
                row_frames.append(horizontal_band)
            row_frames.append(frame)
        combined_row = np.concatenate(row_frames, axis=1)
        if i > 0:
            rows.append(vertical_band)
        rows.append(combined_row)

    final_grid = np.concatenate(rows, axis=0)
    return final_grid

def resize_image_grid(image, max_length=1920):
    width, height = image.size
    if max(width, height) > max_length:
        if width > height:
            scale = max_length / width
        else:
            scale = max_length / height

        new_width = int(width * scale)
        new_height = int(height * scale)

        img_resized = image.resize((new_width, new_height), Image.BILINEAR)
    else:
        img_resized = image
    return img_resized

def load_video_frames_sg4v(video_path, num_segments):
    def get_index(num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)

    frame_indices = get_index(num_frames, num_segments)
    frames = vr.get_batch(frame_indices).asnumpy()

    return frames

## internvl3.5
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video_raw_internvl3_5(video_path, bound=None, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    video_ori = []
    for idx in frame_indices:
        frame = vr[idx].asnumpy()
        video_ori.append(frame)
    video_ori = np.array(video_ori)
    return video_ori

def preprocess_video_internvl3_5(video, input_size=448, max_num=1):
    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)

    for frame in video:
        img = Image.fromarray(frame).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in tiles]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# molmo2 transform frame idx
def apply_transformation_to_frame_idx(frame_idx, transformation, setting):
    T = len(frame_idx)

    if transformation.name == "delete_one_frame":
        drop_i = int(setting)
        new_idx = [frame_idx[j] for j in range(T) if j != drop_i]
        return new_idx

    if transformation.name == "delete_two_frames":
        i = int(setting[0])
        j = int(setting[1])
        new_idx = [frame_idx[k] for k in range(T) if k not in [i, j]]
        return new_idx

    if transformation.name == "reverse":
        return list(reversed(frame_idx))

    if transformation.name == "random_shuffle":
        perm = list(setting)
        return [frame_idx[k] for k in perm]

    if transformation.name == "add_one_frame":
        src_idx = int(setting["add_frame_id"])
        insert_pos = int(setting["insert_position"])
        src_frame = frame_idx[src_idx]

        new_idx = []
        for pos in range(T + 1):
            if pos == insert_pos:
                new_idx.append(src_frame)
            if pos < T:
                new_idx.append(frame_idx[pos])
        return new_idx

    if transformation.name == "add_two_frames":
        src_i = int(setting["add_frame_ids"][0])
        src_j = int(setting["add_frame_ids"][1])
        pos_i = int(setting["insert_positions"][0])
        pos_j = int(setting["insert_positions"][1])

        insert_map = {
            pos_i: frame_idx[src_i],
            pos_j: frame_idx[src_j],
        }

        new_idx = []
        orig_ptr = 0
        for pos in range(T + 2):
            if pos in insert_map:
                new_idx.append(insert_map[pos])
            else:
                new_idx.append(frame_idx[orig_ptr])
                orig_ptr += 1
        return new_idx

    if transformation.name == "replace_one_frame":
        src_idx = int(setting["source_frame_id"])
        tgt_idx = int(setting["target_frame_id"])
        new_idx = list(frame_idx)
        new_idx[tgt_idx] = frame_idx[src_idx]
        return new_idx

    if transformation.name == "replace_two_frames":
        src_i = int(setting["source_frame_ids"][0])
        src_j = int(setting["source_frame_ids"][1])
        tgt_i = int(setting["target_frame_ids"][0])
        tgt_j = int(setting["target_frame_ids"][1])
        new_idx = list(frame_idx)
        new_idx[tgt_i] = frame_idx[src_i]
        new_idx[tgt_j] = frame_idx[src_j]
        return new_idx

    raise ValueError("Unknown transformation name: %s" % transformation.name)
