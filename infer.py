import torch
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
import random

from safetensors.torch import load_file
import gc
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline
)
from trainscripts.textsliders.lora import (
    LoRANetwork,
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
)

import warnings

warnings.filterwarnings("ignore")


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def load_models(pretrained_model_name_or_path, revision, device, weight_dtype):
    noise_scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    # tokenizer = CLIPTokenizer.from_pretrained(
    #     pretrained_model_name_or_path, subfolder="tokenizer", revision=revision
    # )
    # text_encoder = CLIPTextModel.from_pretrained(
    #     pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    # )
    # vae = AutoencoderKL.from_pretrained(
    #     pretrained_model_name_or_path, subfolder="vae", revision=revision
    # )
    # unet = UNet2DConditionModel.from_pretrained(
    #     pretrained_model_name_or_path, subfolder="unet", revision=revision
    # )
    pipe = StableDiffusionPipeline.from_single_file(pretrained_model_name_or_path , torch_dtype = weight_dtype)
    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    # Freeze parameters of models to save more memory
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)

    return noise_scheduler, tokenizer, text_encoder, vae, unet


def generate_images(args, prompts, lora_weight, noise_scheduler, tokenizer, text_encoder, vae, unet, device, weight_dtype):
    scales = [-2,-1, 0, 1, 2, 3, 4]
    start_noise = 700
    num_images_per_prompt = 1

    torch_device = device
    negative_prompt = "naked, nude, ugly, broken hands, long neck"
    batch_size = 1
    height = 768
    width = 512
    ddim_steps = 50
    guidance_scale = 7.5

    os.makedirs(args.out_dir, exist_ok=True)

    for prompt in prompts:
        for _ in range(num_images_per_prompt):
            seed = random.randint(0, int(1e9))

            if "full" in lora_weight:
                train_method = "full"
            elif "noxattn" in lora_weight:
                train_method = "noxattn"
            else:
                train_method = "noxattn"

            network_type = "c3lier"
            if train_method == "xattn":
                network_type = "lierla"
            network_type = "lierla"
            modules = DEFAULT_TARGET_REPLACE
            if network_type == "c3lier":
                modules += UNET_TARGET_REPLACE_MODULE_CONV

            model_name = lora_weight
            name = os.path.basename(model_name)
            os.makedirs(f"{args.out_dir}/{name}", exist_ok=True)

            # unet = UNet2DConditionModel.from_pretrained(
            #     args.model_path, subfolder="unet", revision=args.revision
            # )
            pipe = StableDiffusionPipeline.from_single_file(args.model_path , torch_dtype = weight_dtype)
            unet = pipe.unet
            unet.requires_grad_(False)
            unet.to(device, dtype=weight_dtype)
            del pipe

            rank = 4
            alpha = 1
            if "rank4" in lora_weight:
                rank = 4
            if "rank8" in lora_weight:
                rank = 8
            if "alpha1" in lora_weight:
                alpha = 1.0

            network = LoRANetwork(
                unet,
                rank=rank,
                multiplier=1.0,
                alpha=alpha,
                train_method=train_method,
                target_replace_modules=modules,
            ).to(device, dtype=weight_dtype)

            if lora_weight.endswith(".safetensors"):
                network.load_state_dict(load_file(lora_weight))
            else:
                network.load_state_dict(torch.load(lora_weight))

            images_list = []

            for scale in scales:
                generator = torch.manual_seed(seed)
                text_input = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

                max_length = text_input.input_ids.shape[-1]
                if negative_prompt is None:
                    uncond_input = tokenizer(
                        [""] * batch_size,
                        padding="max_length",
                        max_length=max_length,
                        return_tensors="pt",
                    )
                else:
                    uncond_input = tokenizer(
                        [negative_prompt] * batch_size,
                        padding="max_length",
                        max_length=max_length,
                        return_tensors="pt",
                    )
                uncond_embeddings = text_encoder(
                    uncond_input.input_ids.to(torch_device)
                )[0]

                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

                latents = torch.randn(
                    (batch_size, unet.in_channels, height // 8, width // 8),
                    generator=generator,
                )
                latents = latents.to(torch_device)

                noise_scheduler.set_timesteps(ddim_steps)
                latents = latents * noise_scheduler.init_noise_sigma
                latents = latents.to(weight_dtype)

                for t in noise_scheduler.timesteps:
                    if t > start_noise:
                        network.set_lora_slider(scale=0)
                    else:
                        network.set_lora_slider(scale=scale)

                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = noise_scheduler.scale_model_input(
                        latent_model_input, timestep=t
                    )

                    with network:
                        with torch.no_grad():
                            noise_pred = unet(
                                latent_model_input,
                                t,
                                encoder_hidden_states=text_embeddings,
                            ).sample

                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                    latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

                latents = 1 / 0.18215 * latents
                with torch.no_grad():
                    image = vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                images = (image * 255).round().astype("uint8")
                pil_images = [Image.fromarray(image) for image in images]
                pil_images[0].save(
                    f"{args.out_dir}/{name}/{prompt.replace(' ', '_')}_{scale}_{seed}.png"
                )
                images_list.append(pil_images[0])

            del network, unet
            torch.cuda.empty_cache()
            flush()

            # Create and save the GIF
            gif_path = f"{args.out_dir}/{name}/{prompt.replace(' ', '_')}_{seed}.gif"
            images_list[0].save(
                gif_path,
                save_all=True,
                append_images=images_list[1:],
                duration=500,
                loop=0
            )
            print(f"Saved GIF: {gif_path}")


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Diffusion model with LoRA")
    parser.add_argument("--lora_weight", type=str, required=True, help="Path to the LoRA weight file")
    parser.add_argument("--prompt_file", type=str, default="prompts.txt", help="Path to the prompt file")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--model_path", type=str, default="base_models/veronika", help="Path to the pretrained model")
    parser.add_argument("--revision", type=str, default=None, help="Model revision")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    parser.add_argument("--weight_dtype", type=str, default="float16", help="Weight data type (float16, float32, etc.)")

    args = parser.parse_args()

    # Convert weight_dtype to torch dtype
    weight_dtype = getattr(torch, args.weight_dtype)

    # Read prompts from the file
    with open(args.prompt_file, 'r') as file:
        prompts = [line.strip() for line in file.readlines()]

    noise_scheduler, tokenizer, text_encoder, vae, unet = load_models(args.model_path, args.revision, args.device, weight_dtype)
    generate_images(args, prompts, args.lora_weight, noise_scheduler, tokenizer, text_encoder, vae, unet, args.device, weight_dtype)


if __name__ == "__main__":
    main()
