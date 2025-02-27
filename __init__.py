# Made by https://github.com/blepping

# Usage:
# Put the resolution_normalization.safetensors model in models/unet
# Patch the model with ApplyResAdapterUnet, load the LoRA part normally.

import safetensors
import folder_paths


class StateDictConverter:
    # Modified from comfy.diffusers_convert
    # =================#
    # UNet Conversion #
    # =================#
    unet_conversion_map = (
        # (stable-diffusion, HF Diffusers)
        ("time_embed.0.weight", "time_embedding.linear_1.weight"),
        ("time_embed.0.bias", "time_embedding.linear_1.bias"),
        ("time_embed.2.weight", "time_embedding.linear_2.weight"),
        ("time_embed.2.bias", "time_embedding.linear_2.bias"),
        ("input_blocks.0.0.weight", "conv_in.weight"),
        ("input_blocks.0.0.bias", "conv_in.bias"),
        ("out.0.weight", "conv_norm_out.weight"),
        ("out.0.bias", "conv_norm_out.bias"),
        ("out.2.weight", "conv_out.weight"),
        ("out.2.bias", "conv_out.bias"),
    )

    unet_conversion_map_resnet = (
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0", "norm1"),
        ("in_layers.2", "conv1"),
        ("out_layers.0", "norm2"),
        ("out_layers.3", "conv2"),
        ("emb_layers.1", "time_emb_proj"),
        ("skip_connection", "conv_shortcut"),
    )

    def __init__(self):
        unet_conversion_map_layer = []
        # hardcoded number of downblocks and resnets/attentions...
        # would need smarter logic for other networks.

        for i in range(4):
            # loop over downblocks/upblocks
            for j in range(2):
                # loop over resnets/attentions for downblocks
                hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
                sd_down_res_prefix = f"input_blocks.{3 * i + j + 1}.0."
                unet_conversion_map_layer.append((
                    sd_down_res_prefix,
                    hf_down_res_prefix,
                ))
                if i < 3:
                    # no attention layers in down_blocks.3
                    hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                    sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
                    unet_conversion_map_layer.append((
                        sd_down_atn_prefix,
                        hf_down_atn_prefix,
                    ))
            for j in range(3):
                # loop over resnets/attentions for upblocks
                hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
                sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
                unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))
                if i > 0:
                    # no attention layers in up_blocks.0
                    hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
                    sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
                    unet_conversion_map_layer.append((
                        sd_up_atn_prefix,
                        hf_up_atn_prefix,
                    ))
            if i < 3:
                # no downsample in down_blocks.3
                hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
                sd_downsample_prefix = f"input_blocks.{3 * (i + 1)}.0.op."
                unet_conversion_map_layer.append((
                    sd_downsample_prefix,
                    hf_downsample_prefix,
                ))
                # no upsample in up_blocks.3
                hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
                sd_upsample_prefix = f"output_blocks.{3 * i + 2}.{1 if i == 0 else 2}."
                unet_conversion_map_layer.append((
                    sd_upsample_prefix,
                    hf_upsample_prefix,
                ))
        hf_mid_atn_prefix = "mid_block.attentions.0."
        sd_mid_atn_prefix = "middle_block.1."
        unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))
        for j in range(2):
            hf_mid_res_prefix = f"mid_block.resnets.{j}."
            sd_mid_res_prefix = f"middle_block.{2 * j}."
            unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))
        self.unet_conversion_map_layer = tuple(unet_conversion_map_layer)

    # Modified from comfy.diffusers_convert
    def convert_unet_state_dict(self, unet_state_dict):
        mapping = {k: k for k in unet_state_dict.keys()}
        for sd_name, hf_name in self.unet_conversion_map:
            mapping[hf_name] = sd_name
        for k, v in mapping.items():
            if "resnets" in k:
                for sd_part, hf_part in self.unet_conversion_map_resnet:
                    v = v.replace(hf_part, sd_part)
                mapping[k] = v
        for k, v in mapping.items():
            for sd_part, hf_part in self.unet_conversion_map_layer:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
        new_state_dict = {
            v: unet_state_dict[k] for k, v in mapping.items() if k in unet_state_dict
        }
        return new_state_dict


class ApplyResAdapterUnet:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"

    state_dict_converter = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "unet_name": (folder_paths.get_filename_list("unet"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0}),
            }
        }

    @classmethod
    def load_state_dict(cls, filename):
        with safetensors.safe_open(filename, framework="pt", device="cpu") as fp:
            dsd = {k: fp.get_tensor(k) for k in fp.keys()}
        if cls.state_dict_converter is None:
            cls.state_dict_converter = StateDictConverter()
        return cls.state_dict_converter.convert_unet_state_dict(dsd)

    @classmethod
    def patch(cls, model, unet_name, strength=1.0):
        sd = cls.load_state_dict(folder_paths.get_full_path("unet", unet_name))
        model = model.clone()
        model.add_patches(
            {f"diffusion_model.{k}": (v,) for k, v in sd.items()},
            strength_patch=strength,
            strength_model=min(1.0, max(0.0, 1.0 - strength)),
        )
        return (model,)


NODE_CLASS_MAPPINGS = {"ApplyResAdapterUnet": ApplyResAdapterUnet}
