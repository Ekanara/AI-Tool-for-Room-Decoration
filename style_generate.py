
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


def run_generate_style(prompt, guidance_scale, max_resolution):
    repo_id = "stablediffusionapi/interiordesignsuperm"
    pipeline = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipeline.to("cuda")
    image = pipeline(prompt=prompt, guidance_scale=guidance_scale).images[0]
    return image