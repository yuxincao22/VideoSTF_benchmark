import os
import sys


def set_env_sane_defaults():
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DISABLE_FLASH_ATTENTION", "1")
    os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

def pin_gpu(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

def activate_repo(repo_root):
    """
    Ensure `import llava` resolves to the `llava/` package under `repo_root`.

    repo_root must be the directory that directly contains the `llava/` folder.
    Example:
      .../models/LLaVA-NeXT
      .../models/ShareGPT4Video
    """
    repo_root = os.path.abspath(os.path.expanduser(repo_root))

    # Remove previously imported llava modules to avoid cache pollution
    to_delete = []
    for k in list(sys.modules.keys()):
        if k == "llava" or k.startswith("llava."):
            to_delete.append(k)
    for k in to_delete:
        del sys.modules[k]

    # Put the desired repo root at the front of sys.path
    if repo_root in sys.path:
        sys.path.remove(repo_root)
    sys.path.insert(0, repo_root)
