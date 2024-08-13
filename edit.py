import torch
from PIL import Image
import argparse
import os, json, random
import pandas as pd
import matplotlib.pyplot as plt
import glob, re
import random

from safetensors.torch import load_file
import matplotlib.image as mpimg
import copy
import gc
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler , StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, AttentionProcessor


from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import shutil
from torch.optim.adam import Adam

try:
    os.chdir('sliders')
except:
    pass

import trainscripts.textsliders.ptp_utils as ptp_utils
from trainscripts.textsliders.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).resize((512,640)))[:, :, :3]
        return image
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image
    
LOW_RESOURCE = False 
MAX_NUM_WORDS = 77
weight_dtype = torch.float32 # if you are using GPU >T4 in colab you can use bfloat16
device = 'cuda'
device = torch.device(device)

class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None , guidance_scale = 7.5):
        latents_input = torch.cat([latents] * 2)
        latents_input = latents_input.to(self.model.unet.dtype)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        latents = latents.to(self.model.vae.dtype)
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).to(torch.float16).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device).to(self.model.vae.dtype)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent , ddim_steps = 50 , guidance_scale = 7.5):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        latent = latent.to(self.model.unet.dtype)
        for i in range(ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings )
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image , ddim_steps = 50 , guidance_scale =7.5):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent , ddim_steps = ddim_steps , guidance_scale = guidance_scale)
        
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon , ddim_steps = 50 , guidance_scale = 7.5):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * ddim_steps)
        for i in range(ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context , guidance_scale = guidance_scale)
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False , ddim_steps = 50 , guidance_scale = 7.5):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        
        image_gt = load_512(image_path, *offsets)
        
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt , ddim_steps = ddim_steps , guidance_scale = guidance_scale)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon , ddim_steps = ddim_steps)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        
    
    def __init__(self, model , ddim_steps = 50):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(ddim_steps)
        self.prompt = None
        self.context = None

def concatenate_images(images_list):
    widths, heights = zip(*(img.size for img in images_list))
    total_width = sum(widths)
    max_height = max(heights)

    concatenated_image = Image.new('RGB', (total_width, max_height), "white")
    
    x_offset = 0
    for img in images_list:
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return concatenated_image

def flush():
    torch.cuda.empty_cache()
    gc.collect()

def generate_images(
    image_path,
    prompt,
    lora_weight,
    scales=[-0.5, 0, 1, 2, 3],
    output_path="output.png",
    offsets=(0,0,0,0),
    num_inner_steps=10,
    early_stop_epsilon=1e-5,
    device='cuda',
    weight_dtype=torch.float32,
    pretrained_model_name_or_path="stablediffusionapi/realistic-vision-v51",
    ddim_steps=50,
    guidance_scale=7.5,
    start_noise = 800  # use smaller values for real image editing so that the identity does not change
):
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    if pretrained_model_name_or_path.endswith('.safetensors'):
        ldm_stable = StableDiffusionPipeline.from_single_file(pretrained_model_name_or_path, scheduler=scheduler, torch_dtype=weight_dtype).to(device)
    else :
        ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, scheduler=scheduler, torch_dtype=weight_dtype).to(device)
    
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")

    null_inversion = NullInversion(ldm_stable , ddim_steps = ddim_steps)
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets, num_inner_steps, early_stop_epsilon, verbose=True, ddim_steps = ddim_steps , guidance_scale = guidance_scale)

    uncond_embeddings_copy = copy.deepcopy(uncond_embeddings)

    flush()
    del ldm_stable
    flush()
    text_encoder , tokenizer , unet , vae = None , None , None , None
    noise_scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    print(f'Load pipeline from {pretrained_model_name_or_path}')
    if pretrained_model_name_or_path.endswith('.safetensors'):
        pipe = StableDiffusionPipeline.from_single_file(pretrained_model_name_or_path , torch_dtype =weight_dtype )
        unet = pipe.unet
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
        vae = pipe.vae
    else:
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)

    if "full" in lora_weight:
        train_method = "full"
    elif "noxattn" in lora_weight:
        train_method = "noxattn"
    else:
        train_method = "noxattn"

    network_type = "c3lier"
    if train_method == "xattn":
        network_type = "lierla"
    modules = DEFAULT_TARGET_REPLACE
    if network_type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV
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
    for scale1 in scales:
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings_ = text_encoder(text_input.input_ids.to(device))[0]
        max_length = text_input.input_ids.shape[-1]

        noise_scheduler.set_timesteps(ddim_steps)

        latents = x_t * noise_scheduler.init_noise_sigma
        latents = latents.to(unet.dtype)
        cnt = -1
        for t in tqdm(noise_scheduler.timesteps):
            cnt += 1
            if t > start_noise:
                network.set_lora_slider(scale=0)
            else:
                network.set_lora_slider(scale=scale1)

            text_embeddings = torch.cat([uncond_embeddings_copy[cnt].expand(*text_embeddings_.shape), text_embeddings_])
            latent_model_input = torch.cat([latents] * 2)
            text_embeddings = text_embeddings.to(weight_dtype)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)

            with torch.no_grad():
                with network:
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).to(torch.float16).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        images_list.append(pil_images[0])

    concatenated_image = concatenate_images(images_list)
    concatenated_image.save(output_path)

# Example usage:
generate_images(
    image_path="/workspace/sliders/input/input_2.png",
    prompt="a professional photo of a woman, wearing a bikini, full body",
    lora_weight="/workspace/sliders/models/veronika_dreambooth_prompt_weight_new_env_alpha1.0_rank8_full/veronika_dreambooth_prompt_weight_new_env_alpha1.0_rank8_full_200steps.safetensors",
    output_path="output_start_500_5.png",
    pretrained_model_name_or_path = '/workspace/veronika/veronika.safetensors',
    start_noise = 500 ,
    num_inner_steps = 10 , 
    ddim_steps = 50 , 
    scales=[-0.5, 0, 1, 2, 3],
    early_stop_epsilon=1e-5,
    
)