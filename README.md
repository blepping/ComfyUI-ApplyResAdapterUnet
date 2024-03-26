# ComfyUI-ApplyResAdapterUnet

ComfyUI node to apply the ResAdapter Unet patch for SD1.5 models.

See https://github.com/bytedance/res-adapter for explanation and link to download the LoRA and unet patch.

## Usage

### SDXL

For SDXL, you only need the LoRA (as far as I know) so a dedicated node is unnecessary: just load the LoRA as usual. You don't need this repo.

### SD 1.5

* Put the `resolution_normalization.safetensors` model in `models/unet`
* Patch the model with the `ApplyResAdapterUnet` node, load the `resolution_lora.safetensors` LoRA normally.

You can experiment with different unet and LoRA strengths.
I haven't tested it extensively, but at resolutions above 1024x1024 using full strength doesn't seem to work well (and in fact may be worse than nothing).
It's also possible to combine ResAdapter with other techniques such as Kohya Deep Shrink (AKA `PatchModelAddDownScale`).

## Example Workflow

Workflow with included ComfyUI metadata:

![Example workflow](assets/resadaptercomparisonworkflow.png)

(Simple demonstration, I made no effort to get a pretty picture.)
