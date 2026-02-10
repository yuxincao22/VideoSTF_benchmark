from utils.registry import REGISTRY

adapters = [
    "llava_video_7b_qwen2",
    "llava_video_7b_qwen2_video_only",
    "llava_next_video_7b",
    "llava_next_video_7b_dpo",
    "llava_next_video_32b_qwen",
    "videollama2",
    "sharegpt4video",
    "internvl35_8b",
    "qwen3vl_8b",
    "molmo2_8b",
]

def register_adapters(adapter):
    assert adapter in adapters
    if adapter == "llava_video_7b_qwen2":
        from models.llava_video_7b_qwen2 import LlavaVideo7BQwen2Adapter
        REGISTRY.register(adapter, LlavaVideo7BQwen2Adapter)
    elif adapter == "llava_video_7b_qwen2_video_only":
        from models.llava_video_7b_qwen2_video_only import LlavaVideo7BQwen2VideoOnlyAdapter
        REGISTRY.register(adapter, LlavaVideo7BQwen2VideoOnlyAdapter)
    elif adapter == "llava_next_video_7b":
        from models.llava_next_video_7b import LlavaNextVideo7BAdapter
        REGISTRY.register(adapter, LlavaNextVideo7BAdapter)
    elif adapter == "llava_next_video_7b_dpo":
        from models.llava_next_video_7b_dpo import LlavaNextVideo7BDPOAdapter
        REGISTRY.register(adapter, LlavaNextVideo7BDPOAdapter)
    elif adapter == "llava_next_video_32b_qwen":
        from models.llava_next_video_32b_qwen import LlavaNextVideo32BQwenAdapter
        REGISTRY.register(adapter, LlavaNextVideo32BQwenAdapter)
    elif adapter == "videollama2":
        from models.videollama2 import VideoLLaMA2Adapter
        REGISTRY.register(adapter, VideoLLaMA2Adapter)
    elif adapter == "sharegpt4video":
        from models.sharegpt4video import ShareGPT4VideoAdapter
        REGISTRY.register(adapter, ShareGPT4VideoAdapter)
    elif adapter == "internvl35_8b":
        from models.internvl35_8b import InternVL358BAdapter
        REGISTRY.register(adapter, InternVL358BAdapter)
    elif adapter == "qwen3vl_8b":
        from models.qwen3vl_8b import Qwen3VL8BAdapter
        REGISTRY.register(adapter, Qwen3VL8BAdapter)
    elif adapter == "molmo2_8b":
        from models.molmo2_8b import Molmo28BAdapter
        REGISTRY.register(adapter, Molmo28BAdapter)