# Made by https://github.com/blepping

# Usage:
# Put the resolution_normalization.safetensors model in models/unet
# Patch the model with ApplyResAdapterUnet, load the LoRA part normally.

import safetensors
from comfy.diffusers_convert import (
    unet_conversion_map,
    unet_conversion_map_resnet,
    unet_conversion_map_layer,
)
import folder_paths


# Modified from comfy.diffusers_convert
def convert_unet_state_dict(unet_state_dict):
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    for k, v in mapping.items():
        if "resnets" in k:
            for sd_part, hf_part in unet_conversion_map_resnet:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        for sd_part, hf_part in unet_conversion_map_layer:
            v = v.replace(hf_part, sd_part)
        mapping[k] = v
    new_state_dict = {
        v: unet_state_dict[k] for k, v in mapping.items() if k in unet_state_dict
    }
    return new_state_dict


def load_state_dict(fn):
    with safetensors.safe_open(fn, framework="pt", device="cpu") as fp:
        dsd = {k: fp.get_tensor(k) for k in fp.keys()}
    return convert_unet_state_dict(dsd)


class ApplyResAdapterUnet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "unet_name": (folder_paths.get_filename_list("unet"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"

    def patch(self, model, unet_name, strength=1.0):
        sd = load_state_dict(folder_paths.get_full_path("unet", unet_name))
        model = model.clone()
        model.add_patches(
            {f"diffusion_model.{k}": (v,) for k, v in sd.items()},
            strength_patch=strength,
            strength_model=min(1.0, max(0.0, 1.0 - strength)),
        )
        return (model,)


NODE_CLASS_MAPPINGS = {"ApplyResAdapterUnet": ApplyResAdapterUnet}
