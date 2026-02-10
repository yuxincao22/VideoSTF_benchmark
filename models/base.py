class BaseAdapter:
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.n = 5
        self.threshold = 1

    def load(self):
        raise NotImplementedError

    def generate(self, model_input, prompt):
        raise NotImplementedError

    def stress_test(self, video_path, video_name, csv_path, transformations, prompt):
        raise NotImplementedError

    def attack(self, video_path, video_name, csv_path, transformations, prompt, finished_attacks):
        raise NotImplementedError