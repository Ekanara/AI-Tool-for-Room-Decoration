
import requests
from PIL import Image
import io
import random
import json
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import DDIMScheduler
import torch
from src.freeU.free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d


def run_generate_style_nofreeu(prompt, negative_prompt, guidance_scale, height, width):
    repo_id = "stablediffusionapi/interiordesignsuperm"
    pipeline = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipeline.to("cuda")

    image = pipeline(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps=50, height=height, width=width, negative_prompt=negative_prompt).images[0]

    #upscaled_image = image.resize((max_resolution, max_resolution), Image.LANCZOS)  # Resize with anti-aliasing
    return image