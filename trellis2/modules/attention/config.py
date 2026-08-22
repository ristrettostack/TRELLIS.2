from typing import *

BACKEND = 'flash_attn' 
DEBUG = False

_ATTN_BACKENDS = ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']


def _gpu_supports_flash_attn() -> bool:
    """
    flash_attn / flash_attn_3's published wheels and fused kernels target sm80+
    (Ampere, Ada, Hopper). Turing cards like the T4 (sm75) aren't supported - the
    package either fails to import, or the kernel errors out at call time deep inside
    a forward pass. `sdpa` (PyTorch's built-in fused attention) runs fine in fp16 on
    Turing and is the safe fallback there.
    """
    import torch
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def _resolve_backend(requested: str, *, explicit: bool) -> str:
    if requested in ('flash_attn', 'flash_attn_3') and not _gpu_supports_flash_attn():
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'
        except Exception:
            gpu_name = 'unknown GPU'
        # Prefer xformers if it's installed: it has an officially supported Turing
        # (sm75+, e.g. T4) kernel path, and PyTorch's own `sdpa` actually dispatches
        # to xformers' memory-efficient kernel under the hood on Turing anyway when
        # it's available - so using it directly is at least as fast and more
        # predictable than relying on sdpa's internal backend selection.
        try:
            import xformers.ops  # noqa: F401
            fallback = 'xformers'
        except ImportError:
            fallback = 'sdpa'
        print(
            f"[ATTENTION] '{requested}' was {'explicitly requested' if explicit else 'the default'} "
            f"but is not supported on this GPU ({gpu_name}, compute capability < 8.0). "
            f"Falling back to '{fallback}'."
        )
        return fallback
    return requested


def __from_env():
    import os
    
    global BACKEND
    global DEBUG
    
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_attn_debug = os.environ.get('ATTN_DEBUG')
    
    explicit = env_attn_backend is not None and env_attn_backend in _ATTN_BACKENDS
    if explicit:
        BACKEND = env_attn_backend
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'

    BACKEND = _resolve_backend(BACKEND, explicit=explicit)

    print(f"[ATTENTION] Using backend: {BACKEND}")
        

__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']):
    global BACKEND
    BACKEND = _resolve_backend(backend, explicit=True)

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug