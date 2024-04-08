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
prompts = [
    "Japan Style, Sunlight filtering through shoji screens, minimalist wooden furniture, a single bonsai tree in a ceramic pot, the subtle scent of tatami mats."

    "Japan Style, Low platform on smooth wooden floors, plush cushions and a single painted scroll on the wall, soft natural light creating a sense of tranquil elegance."
    
    "Japan Style, Open space with natural elements, bamboo accents, the gentle sound of a rock garden fountain, a touch of vibrant color in a single piece of artwork."
    
    "Luxury Western, Grand space with high ceilings, crystal chandeliers casting warm light, plush velvet upholstery, gilded accents on antique furniture."
    
    "Chinese interior, bedroom, red silk curtains with dragon and phoenix embroidery, paper lanterns, antique porcelain, lacquered wood dressing table. (Feels warm and traditional)"
    
    "Indochine Style, Opulent fabrics like satin and brocade, mirrored surfaces reflecting light, a crystal decanter on a silver tray, a grand piano in the corner."
    
    "Indochine Style, Dark wood furniture with rattan accents, vibrant tropical fabrics, patterned tile floors, the scent of sandalwood incense."
    
    "Indochine Style, Lush potted palms, sunlight filtering through woven blinds, antique Chinese porcelain, weathered wood with layers of history."
    
    "Colonial elegance meets tropical warmth, French windows open to a vibrant garden, vintage travel posters, and the faint sound of a ceiling fan."
    
    "Parisian Style, Sunlight streaming through tall windows with wrought iron balconies, herringbone wood floors, molding on the walls, a bouquet of fresh flowers on a marble side table. "
    
    "Parisian Style ,Antique furniture with elegant curves, worn velvet cushions, a collection of vintage books and photographs, a crystal chandelier casting soft light."
    
    "Parisian Style, A mix of old and new, modern artwork against a backdrop of historical architecture, a cozy reading nook with a plush armchair and overflowing bookshelves."
]
repo_id = "stablediffusionapi/interiordesignsuperm"
pipeline = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipeline.to("cuda")
nofreeu_images = pipeline(prompt=prompts, guidance_scale=3.5, num_inference_steps=50, num_images_per_prompt= 210, output_type="np").images

# register_free_upblock2d(pipeline, b1= 1.5, b2= 1.6, s1 = 0.9, s2 = 0.2)
# register_free_crossattn_upblock2d(pipeline, b1= 1.5, b2= 1.6, s1 = 0.9, s2 = 0.2)
register_free_upblock2d(pipeline, b1=1.3, b2=1.5, s1=0.9, s2=0.2)
register_free_crossattn_upblock2d(pipeline, b1=1.3, b2=1.5, s1=0.9, s2=0.2)


from torchmetrics.functional.multimodal import clip_score
from functools import partial

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

freeu_images = pipeline(prompt=prompts, guidance_scale=3.5, num_inference_steps=50, num_images_per_prompt= 210, output_type="np").images


freeu_clip_score = calculate_clip_score(freeu_images, prompts)
print(f"FreeU CLIP score: {freeu_clip_score}")

nofreeu_clip_score = calculate_clip_score(nofreeu_images, prompts)
print(f"No FreeU CLIP score: {nofreeu_clip_score}")

