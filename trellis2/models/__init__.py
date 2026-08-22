import importlib
from typing import Optional

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def _gpu_supports_bf16() -> bool:
    """
    True only on GPUs with hardware-accelerated bf16 tensor cores (Ampere+, sm_80+).
    Turing cards (T4, sm_75) and older report False - bf16 either isn't supported
    or falls back to slow, unaccelerated kernels on those devices.
    """
    import torch
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def from_pretrained(path: str, force_fp16: Optional[bool] = None, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        force_fp16: If True, override whatever dtype/use_fp16 is baked into the checkpoint's config
              (e.g. 'bfloat16', which many of the gen configs default to) and load the model natively
              in float16 instead. This matters on Turing GPUs like the T4, which have no accelerated
              bf16 tensor-core path - bf16 there is either unsupported or extremely slow, while fp16
              runs at full tensor-core speed. If left as None, this is auto-detected: fp16 is forced
              automatically when a bf16-unfriendly GPU (compute capability < 8.0, e.g. T4) is detected.
        **kwargs: Additional arguments for the model constructor. These take precedence over anything
              of the same name found in the checkpoint's config['args'].
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)

    args = dict(config['args'])

    if force_fp16 is None:
        force_fp16 = not _gpu_supports_bf16()

    if force_fp16:
        # Flow models (sparse_structure_flow / structured_latent_flow) take a string `dtype` arg.
        if 'dtype' in args:
            if args['dtype'] not in ('float16', 'fp16'):
                print(f"[trellis2] Overriding model dtype '{args['dtype']}' -> 'float16' "
                      f"(force_fp16=True or non-bf16-capable GPU detected).")
            args['dtype'] = 'float16'
        # VAE models (sparse_structure_vae / sc_vaes) take a bool `use_fp16` arg.
        if 'use_fp16' in args:
            args['use_fp16'] = True

    args.update(kwargs)  # explicit kwargs always win

    # NOTE: we deliberately do NOT call model.half() on the whole module here. Both the flow
    # models (convert_to) and the VAE models (convert_to_fp16) only cast their transformer/conv
    # "torso" (self.blocks / self.middle_block) to the target dtype, and keep input_layer /
    # out_layer / t_embedder / adaLN_modulation in fp32 on purpose - forward() feeds those layers
    # tensors in the *caller's* dtype (h.type(self.dtype) / h.type(x.dtype)) and relies on them
    # staying fp32. Force-halving the whole model would break that and crash with a dtype
    # mismatch. Setting dtype='float16'/use_fp16=True above is enough: it makes the torso (the
    # overwhelming majority of the FLOPs) run in fp16, which is exactly what's accelerated on the
    # T4's tensor cores, while the small I/O layers safely stay fp32.
    model = __getattr__(config['name'])(**args)
    model.load_state_dict(load_file(model_file), strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder