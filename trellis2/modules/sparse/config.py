from typing import *

CONV = 'flex_gemm' 
DEBUG = False
ATTN = 'flash_attn'

_SPARSE_ATTN_BACKENDS = ['xformers', 'flash_attn', 'flash_attn_3']


def _gpu_supports_flash_attn() -> bool:
    """
    flash_attn / flash_attn_3's published wheels and fused kernels target sm80+
    (Ampere, Ada, Hopper). Turing cards like the T4 (sm75) aren't supported - the
    package either fails to import, or the kernel errors out at call time.
    """
    import torch
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def _resolve_attn_backend(requested: str, *, explicit: bool) -> str:
    if requested not in ('flash_attn', 'flash_attn_3') or _gpu_supports_flash_attn():
        return requested

    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'
    except Exception:
        gpu_name = 'unknown GPU'

    # NOTE: unlike modules/attention/config.py (the dense attention path), this
    # sparse attention module has no 'sdpa'/'naive' implementation - full_attn.py
    # and windowed_attn.py only branch on 'xformers' / 'flash_attn' / 'flash_attn_3'.
    # So on an unsupported GPU, xformers is the *only* working fallback.
    try:
        import xformers.ops  # noqa: F401
        print(
            f"[SPARSE] '{requested}' was {'explicitly requested' if explicit else 'the default'} "
            f"but is not supported on this GPU ({gpu_name}, compute capability < 8.0). "
            f"Falling back to 'xformers'."
        )
        return 'xformers'
    except ImportError:
        print(
            f"[SPARSE] WARNING: '{requested}' was {'explicitly requested' if explicit else 'the default'} "
            f"but is not supported on this GPU ({gpu_name}, compute capability < 8.0), and xformers "
            f"(the only other implemented sparse attention backend) is not installed. Sparse attention "
            f"will fail at runtime. Install xformers (see setup.sh --xformers) or set "
            f"SPARSE_ATTN_BACKEND=xformers after installing it."
        )
        return requested  # leave as-is; there is nothing safe to fall back to


def __from_env():
    import os
    
    global CONV
    global DEBUG
    global ATTN
    
    env_sparse_conv_backend = os.environ.get('SPARSE_CONV_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn_backend is None:
        env_sparse_attn_backend = os.environ.get('ATTN_BACKEND')

    explicit_attn = env_sparse_attn_backend is not None and env_sparse_attn_backend in _SPARSE_ATTN_BACKENDS

    if env_sparse_conv_backend is not None and env_sparse_conv_backend in ['none', 'spconv', 'torchsparse', 'flex_gemm']:
        CONV = env_sparse_conv_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if explicit_attn:
        ATTN = env_sparse_attn_backend

    ATTN = _resolve_attn_backend(ATTN, explicit=explicit_attn)

    print(f"[SPARSE] Conv backend: {CONV}; Attention backend: {ATTN}")
        

__from_env()
    

def set_conv_backend(backend: Literal['none', 'spconv', 'torchsparse', 'flex_gemm']):
    global CONV
    CONV = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3']):
    global ATTN
    ATTN = _resolve_attn_backend(backend, explicit=True)