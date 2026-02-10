import copy
import random

import numpy as np

class TemporalTransformation:
    name = None

    def apply(self, video):
        """
        Input: video, shape = [T, ...]
        Output:
            transformed_video
            transformation_setting
        """
        raise NotImplementedError


class AddOneFrame(TemporalTransformation):
    name = "add_one_frame"

    def __init__(self, times=10):
        self.times = times

    def apply(self, video):
        T = len(video)
        results = []

        for _ in range(self.times):
            src_idx = random.randrange(T)
            insert_pos = random.randrange(T + 1)

            new_video = []
            for i in range(T + 1):
                if i == insert_pos:
                    new_video.append(video[src_idx])
                if i < T:
                    new_video.append(video[i])
            new_video = np.array(new_video)
            results.append((
                new_video,
                {
                    "add_frame_id": src_idx,
                    "insert_position": insert_pos
                }
            ))

        return results


class AddTwoFrames(TemporalTransformation):
    name = "add_two_frames"

    def __init__(self, times=10):
        self.times = times

    def apply(self, video):
        T = len(video)
        results = []

        for _ in range(self.times):
            src_i, src_j = random.sample(range(T), 2)
            pos_i, pos_j = random.sample(range(T + 2), 2)
            insert_map = {
                pos_i: video[src_i],
                pos_j: video[src_j]
            }

            new_video = []
            orig_ptr = 0
            for i in range(T + 2):
                if i in insert_map:
                    new_video.append(insert_map[i])
                else:
                    new_video.append(video[orig_ptr])
                    orig_ptr += 1
            new_video = np.array(new_video)
            results.append((
                new_video,
                {
                    "add_frame_ids": [src_i, src_j],
                    "insert_positions": [pos_i, pos_j]
                }
            ))

        return results


class DeleteOneFrame(TemporalTransformation):
    name = "delete_one_frame"

    def apply(self, video):
        T = len(video)
        results = []
        for drop_i in range(T):
            idx = [j for j in range(T) if j != drop_i]
            results.append((
                video[idx],
                drop_i
            ))
        return results


class DeleteTwoFrames(TemporalTransformation):
    name = "delete_two_frames"
    def __init__(self, times=30):
        self.times = times

    def apply(self, video):
        T = len(video)
        results = []
        for _ in range(self.times):
            i, j = random.sample(range(T), 2)
            idx = [k for k in range(T) if k not in [i, j]]
            results.append((
                video[idx],
                [i, j]
            ))
        return results


class ReplaceOneFrame(TemporalTransformation):
    name = "replace_one_frame"

    def __init__(self, times=10):
        self.times = times

    def apply(self, video):
        T = len(video)
        results = []

        for _ in range(self.times):
            src_idx = random.randrange(T)
            tgt_idx = random.choice([i for i in range(T) if i != src_idx])

            new_video = copy.deepcopy(video)
            new_video[tgt_idx] = video[src_idx]

            results.append((
                new_video,
                {
                    "source_frame_id": src_idx,
                    "target_frame_id": tgt_idx
                }
            ))

        return results


class ReplaceTwoFrames(TemporalTransformation):
    name = "replace_two_frames"

    def __init__(self, times=10):
        self.times = times

    def apply(self, video):
        T = len(video)
        results = []

        for _ in range(self.times):
            src_i, src_j = random.sample(range(T), 2)
            candidates = [i for i in range(T) if i not in [src_i, src_j]]
            tgt_i, tgt_j = random.sample(candidates, 2)

            new_video = copy.deepcopy(video)
            new_video[tgt_i] = video[src_i]
            new_video[tgt_j] = video[src_j]

            results.append((
                new_video,
                {
                    "source_frame_ids": [src_i, src_j],
                    "target_frame_ids": [tgt_i, tgt_j]
                }
            ))

        return results


class ReverseVideo(TemporalTransformation):
    name = "reverse"

    def apply(self, video):
        idx = list(reversed(range(len(video))))
        return [(video[idx], "reverse")]


class RandomShuffle(TemporalTransformation):
    name = "random_shuffle"

    def __init__(self, times=10):
        self.times = times

    def apply(self, video):
        T = len(video)
        results = []
        for _ in range(self.times):
            idx = list(range(T))
            random.shuffle(idx)
            results.append((
                video[idx],
                idx
            ))
        return results