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
import gc
from src.freeU.free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
import numpy as np
prompts = [  "Minimalist: Clean lines, neutral colors, and open spaces create a sense of simplicity and tranquility.",
           "Bohemian: Eclectic mix of colors, patterns, and textures, with a cozy and relaxed atmosphere.",
           "Industrial: Exposed brick, metal accents, and raw materials give a rugged and urban feel.",
           "Scandinavian: Light, airy spaces with natural elements, simple furniture, and a focus on functionality.",
           "Mediterranean: Warm colors, rustic textures, and natural light evoke the feel of coastal regions.",
           "Mid-Century Modern: Retro-inspired design with sleek furniture, geometric patterns, and a timeless appeal.",
           "Contemporary: Clean lines, minimal clutter, and a mix of modern and traditional elements for a sophisticated look.",
           "Farmhouse: Rustic charm with reclaimed wood, vintage decor, and cozy accents for a homey feel.",
           "Art Deco: Glamorous and luxurious, with bold geometric patterns, rich colors, and lavish materials.",
           "Eclectic: A mix of styles, eras, and cultures, with a focus on personal expression and creativity.",  "French Country: Elegant yet cozy, with soft colors, floral patterns, and rustic furniture.",  "Japanese: Serene and minimalist, with natural materials, sliding doors, and a connection to nature.",  "Victorian: Ornate furnishings, rich colors, and intricate details for a sense of opulence and grandeur.",  "Transitional: A blend of traditional and contemporary styles, with clean lines and comfortable furnishings.",  "Rustic: Natural materials, earthy colors, and a cozy, lived-in feel.",  "Gothic: Dramatic and moody, with dark colors, ornate details, and a sense of mystery.",  "Art Nouveau: Organic shapes, floral motifs, and decorative elements inspired by nature.",  "Tropical: Bright colors, lush greenery, and natural textures create a vacation-like paradise.",  "Shabby Chic: Vintage-inspired, with distressed furniture, soft colors, and feminine touches.",  "Zen: Clean, uncluttered spaces designed to promote relaxation and mindfulness.",  "Coastal: Light, breezy colors, natural materials, and nautical decor evoke the feel of the seaside.",  "Southwestern: Warm hues, Native American patterns, and rustic elements reflect the desert landscape.",  "Modern Farmhouse: A mix of rustic and modern elements, with clean lines and cozy accents.",  "Nautical: Navy and white color scheme, striped patterns, and maritime decor for a coastal vibe.",  "Cottage: Quaint and cozy, with vintage furnishings, floral prints, and a lived-in feel.",  "Tuscan: Warm, earthy colors, textured walls, and rustic furniture inspired by the Italian countryside.",  "Moroccan: Vibrant colors, intricate patterns, and rich textures create an exotic and luxurious atmosphere.",  "Hollywood Regency: Glamorous and sophisticated, with bold colors, mirrored surfaces, and luxe fabrics.",  "Bauhaus: Functional yet stylish, with clean lines, geometric shapes, and a focus on form following function.",  "Retro: Nostalgic design inspired by the past, with bold colors, funky patterns, and vintage furniture.",  "Renaissance: Ornate details, rich fabrics, and classical motifs for a sense of elegance and grandeur.",  "Southwestern: Warm hues, Native American patterns, and rustic elements reflect the desert landscape.",  "Traditional: Classic and timeless, with elegant furnishings, rich colors, and ornate details.",  "Asian: Serene and minimalist, with natural materials, clean lines, and a focus on balance and harmony.",  "English Country: Cozy and charming, with floral patterns, antique furniture, and a lived-in feel.",  "Urban Modern: Sleek, industrial-inspired design with concrete floors, exposed ductwork, and minimalist furniture.",  "Boho Chic: Relaxed and eclectic, with a mix of colors, patterns, and textures inspired by travel and culture.",  "Rococo: Ornate and extravagant, with elaborate decorations, pastel colors, and gilded accents.",  "Neoclassical: Inspired by ancient Greek and Roman design, with symmetry, columns, and classical motifs.",  "Southwestern: Warm hues, Native American patterns, and rustic elements reflect the desert landscape.",  "Shaker: Simple and functional, with clean lines, minimal decoration, and handmade furniture.",  "Colonial: Classic and elegant, with traditional furnishings, muted colors, and refined details.",  "Artisanal: Handcrafted furniture, natural materials, and unique, one-of-a-kind pieces for a personalized touch.",  "Scandinavian: Light, airy spaces with natural elements, simple furniture, and a focus on functionality.",  "French Provincial: Elegant and romantic, with distressed wood, soft colors, and ornate details.",  "Modern Gothic: Dark colors, dramatic contrasts, and Gothic-inspired architecture for a contemporary twist.",  "Moorish: Intricate tilework, arched doorways, and vibrant colors create an exotic and luxurious atmosphere.",  "Global: A fusion of cultures and styles from around the world, with bold colors, eclectic furnishings, and unique artifacts.",  "Regency: Elegant and refined, with neoclassical influences, luxurious fabrics, and decorative details.",  "Shaker: Simple and functional, with clean lines, minimal decoration, and handmade furniture.",  "Colonial: Classic and elegant, with traditional furnishings, muted colors, and refined details.",  "Artisanal: Handcrafted furniture, natural materials, and unique, one-of-a-kind pieces for a personalized touch.",  "Scandinavian: Light, airy spaces with natural elements, simple furniture, and a focus on functionality.",  "French Provincial: Elegant and romantic, with distressed wood, soft colors, and ornate details.",  "Modern Gothic: Dark colors, dramatic contrasts, and Gothic-inspired architecture for a contemporary twist.",  "Moorish: Intricate tilework, arched doorways, and vibrant colors create an exotic and luxurious atmosphere.",  "Global: A fusion of cultures and styles from around the world, with bold colors, eclectic furnishings, and unique artifacts.",  "Regency: Elegant and refined, with neoclassical influences, luxurious fabrics, and decorative details.",  "Scandinavian: Light, airy spaces with natural elements, simple furniture, and a focus on functionality.",  "French Provincial: Elegant and romantic, with distressed wood, soft colors, and ornate details.",  "Modern Gothic: Dark colors, dramatic contrasts, and Gothic-inspired architecture for a contemporary twist.",  "Moorish: Intricate tilework, arched doorways, and vibrant colors create an exotic and luxurious atmosphere.",  "Global: A fusion of cultures and styles from around the world, with bold colors, eclectic furnishings, and unique artifacts.",  "Regency: Elegant and refined, with neoclassical influences, luxurious fabrics, and decorative details.",  "Cape Cod: Casual and coastal-inspired, with light colors, natural materials, and nautical accents.",  "Bali: Balinese-inspired design with tropical foliage, rich textures, and traditional craftsmanship.",  "Artisanal: Handcrafted furniture, natural materials, and unique, one-of-a-kind pieces for a personalized touch.",  "Modern Gothic: Dark colors, dramatic contrasts, and Gothic-inspired architecture for a contemporary twist.",  "Moorish: Intricate tilework, arched doorways, and vibrant colors create an exotic and luxurious atmosphere.",  "Global: A fusion of cultures and styles from around the world, with bold colors, eclectic furnishings, and unique artifacts.",  "Regency: Elegant and refined, with neoclassical influences, luxurious fabrics, and decorative details.",  "Cape Cod: Casual and coastal-inspired, with light colors, natural materials, and nautical accents.",  "Bali: Balinese-inspired design with tropical foliage, rich textures, and traditional craftsmanship.",  "Mission: Simple and functional, with clean lines, wood furniture, and traditional craftsmanship.",  "Swedish: Light colors, natural materials, and minimalist design for a fresh and airy feel.",  "Spanish Colonial: Rustic elegance with wrought iron accents, terra cotta tiles, and vibrant colors.",  "Artisanal: Handcrafted furniture, natural materials, and unique, one-of-a-kind pieces for a personalized touch.",  "Modern Gothic: Dark colors, dramatic contrasts, and Gothic-inspired architecture for a contemporary twist.",  "Moorish: Intricate tilework, arched doorways, and vibrant colors create an exotic and luxurious atmosphere.",  "Global: A fusion of cultures and styles from around the world, with bold colors, eclectic furnishings, and unique artifacts.",  "Regency: Elegant and refined, with neoclassical influences, luxurious fabrics, and decorative details.",  "Cape Cod: Casual and coastal-inspired, with light colors, natural materials, and nautical accents.",  "Bali: Balinese-inspired design with tropical foliage, rich textures, and traditional craftsmanship.",  "Mission: Simple and functional, with clean lines, wood furniture, and traditional craftsmanship.",  "Swedish: Light colors, natural materials, and minimalist design for a fresh and airy feel.",  "Spanish Colonial: Rustic elegance with wrought iron accents, terra cotta tiles, and vibrant colors."
          "Italian Renaissance: Living Room, Ornate sofa set, carved wooden coffee table, gilded mirrors, velvet armchairs, elaborate chandeliers, marble fireplace.",
  "Georgian: Dining Room, Mahogany dining table with upholstered chairs, sideboard with china cabinet, crystal chandelier, Georgian-style mirror, antique rug.",
  "Hollywood Glam: Bedroom, Tufted velvet bed frame, mirrored nightstands, vanity with Hollywood lights, plush area rug, crystal table lamps, faux fur throw pillows.",
  "Urban Industrial: Office, Metal and wood desk, industrial shelving units, vintage leather armchair, Edison bulb desk lamp, metal file cabinet, exposed brick walls.",
  "French Art Deco: Lounge, Art Deco sofa with geometric patterns, lacquered coffee table, mirrored accent pieces, velvet chaise lounge, sculptural floor lamp, Deco-inspired rug.",
  "Boho Minimalist: Study, Low-profile platform desk, floor cushions, minimalist bookshelves, woven rattan chair, hanging pendant lights, neutral area rug.",
  "Contemporary Coastal: Sunroom, Wicker furniture set with white cushions, rattan coffee table, indoor plants, sheer curtains, ocean-themed artwork, sisal rug.",
  "Organic Modern: Conservatory, Natural wood dining table with modern chairs, live-edge coffee table, indoor potted plants, rattan hanging chair, bamboo blinds, jute rug.",
  "Desert Modern: Patio, Outdoor sectional with weather-resistant cushions, concrete fire pit, succulent planters, rattan lounge chairs, shade sails, terra cotta floor tiles.",
  "Zen Minimalism: Meditation Room, Low platform meditation cushion, Japanese shoji screen, bamboo floor mat, minimalist altar, incense holder, Zen garden.",
  "Vintage Industrial: Loft, Reclaimed wood dining table, metal bar stools, industrial pendant lights, leather sofa, vintage metal locker storage, exposed ductwork.",
  "Coastal Farmhouse: Breakfast Nook, Farmhouse-style dining table with bench seating, Windsor chairs, pendant light fixture with rope accents, rustic wooden shelves, coastal artwork.",
  "Scandinavian Modern: Family Room, Modular sofa with clean lines, Scandinavian armchair, minimalist coffee table, plush area rug, floor-to-ceiling windows, light wood flooring.",
  "Industrial Chic: Bar, Repurposed wood bar counter, metal bar stools, exposed bulb pendant lights, brick accent wall, industrial shelving for bottles and glasses.",
  "Coastal Contemporary: Game Room, Sleek sectional sofa, gaming table with modern chairs, wall-mounted TV, colorful accent rug, coastal-themed wall art, built-in bookshelves.",
  "Urban Loft: Studio Apartment, Multifunctional furniture pieces, industrial-style lighting fixtures, exposed pipes, concrete floors, lofted bed with storage underneath, floor-to-ceiling windows.",
  "Modern Mediterranean: Formal Dining Room, Large wooden dining table, upholstered chairs with nailhead trim, wrought iron chandelier, terracotta tile flooring, arched windows with flowing curtains.",
  "California Casual: Outdoor Kitchen, Stainless steel appliances, granite countertops, outdoor dining set, built-in grill with stone surround, pergola with retractable canopy, outdoor refrigerator.",
  "Industrial Rustic: Workshop, Wooden workbench with metal legs, industrial shelving for tools, pegboard wall storage, task lighting, concrete floors, rugged stools for seating.",
  "Beach Cottage: Beach House, White slipcovered furniture, distressed wood coffee table, nautical accents, sisal rugs, breezy curtains, beach-themed artwork.",
  "Artistic Bohemian: Art Studio, Boho-inspired furniture, colorful rugs, eclectic decor, easel with canvas, natural light, storage for art supplies.",
  "Vintage Glamour: Dressing Room, Ornate vanity table with mirror, velvet pouf, crystal chandelier, antique wardrobe, Hollywood-style dressing screen, plush area rug.",
  "Rustic Chic: Cabin, Wood-paneled walls, stone fireplace, distressed leather furniture, plaid accents, antler chandelier, cozy throw blankets.",
  "Coastal Chic: Beach Retreat, White slipcovered furniture, seafoam green accents, driftwood decor, seashell accessories, jute rugs, ocean views.",
  "Industrial Farmhouse: Barn Conversion, Rustic farmhouse table, metal dining chairs, sliding barn doors, exposed beams, vintage-inspired lighting fixtures, shiplap walls.",
  "Classic Contemporary: Formal Sitting Room, Neutral color palette, elegant furnishings, statement lighting, tailored drapery, abstract artwork, plush area rug.",
  "Retro Futurism: Retro-themed Room, Retro-inspired furniture, bold colors, futuristic accents, geometric patterns, space-age lighting fixtures.",
  "Bohemian Rhapsody: Music Room, Colorful rugs, floor cushions, eclectic decor, instruments on display, tapestries, cozy seating area.",
  "Desert Chic: Oasis Room, Neutral color palette with pops of terracotta, rattan furniture, cactus and succulent plants, Southwestern textiles, natural materials.",
  "Global Eclectic: Worldly Lounge, Mix of patterns and textures, eclectic decor from around the world, vibrant colors, ethnic-inspired textiles, global artifacts.",
  "Contemporary Rustic: Lodge, Reclaimed wood furniture, stone fireplace, leather seating, antler chandelier, cozy throws, nature-inspired accents.",
  "Urban Bohemian: Loft Apartment, Mix of vintage and modern furniture, eclectic decor, colorful textiles, indoor plants, art gallery wall.",
  "Industrial Vintage: Vintage Storefront, Vintage display cases, industrial shelving, reclaimed wood counters, Edison bulb lighting, exposed brick walls.",
  "Coastal Industrial: Beachfront Warehouse, Exposed pipes, concrete floors, industrial pendant lights, distressed wood furniture, nautical accents, expansive windows.",
  "Traditional Glam: Victorian Parlor, Ornate furnishings, velvet upholstery, crystal chandelier, intricate moldings, Persian rug, grand piano.",
  "Modern Boho: Urban Loft, Layered textiles, indoor plants, macrame accents, rattan furniture, colorful rugs, cozy seating areas.",
  "Scandinavian Chic: Winter Cabin, Neutral color palette, cozy textiles, minimalist decor, sheepskin rugs, natural wood accents, large windows with forest views.",
  "Coastal Modern: Coastal Condo, Clean lines, light wood furniture, seafoam green accents, panoramic ocean views, contemporary artwork.",
  "Rustic Industrial: Industrial Loft, Exposed pipes, concrete floors, metal furniture, reclaimed wood accents, pendant lighting, leather seating.",
  "Boho Glam: Boudoir, Luxurious textiles, mirrored furniture, plush bedding, crystal chandelier, bold patterns, eclectic decor.",
  "Modern Classic: Modern Mansion, Timeless furnishings, neutral color palette, statement artwork, marble accents, sleek lines, grand staircase.",
  "Coastal Boho: Boho Beach House, Bohemian-inspired decor, natural materials, rattan furniture, surf-themed accents, ocean views, relaxed atmosphere.",
  "Urban Boho Chic: Urban Loft, Eclectic mix of furniture styles, bold patterns, vibrant colors, vintage accents, indoor plants, relaxed vibe.",
  "Industrial Modern: Industrial Loft, Minimalist furniture, industrial accents, concrete floors, open floor plan, exposed ductwork, floor-to-ceiling windows.",
  "Classic Coastal: Classic Beach House, Timeless coastal decor, blue and white color palette, wicker furniture, striped accents, seashell decor, oceanfront views.",
  "Boho Modern: Boho Loft, Modern furniture with bohemian accents, eclectic decor, colorful textiles, indoor plants, mix of textures.",
  "Rustic Modern: Modern Ranch House, Blend of rustic and contemporary styles, clean lines, natural materials, neutral color palette, expansive windows, open floor plan.",
  "Eclectic Bohemian: Eclectic Loft, Mix of colors, patterns, and textures, vintage and modern furniture, global-inspired decor, eclectic artwork, vibrant atmosphere.",
  "Mid-Century Eclectic: Mid-Century Modern House, Iconic mid-century furniture pieces, bold patterns, vibrant colors, retro accents, atomic-inspired decor."
]

from torchmetrics.functional.multimodal import clip_score
from functools import partial

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    if images_int.ndim == 3:
        # Single image case
        images_tensor = torch.from_numpy(images_int).permute(2, 0, 1).unsqueeze(0)
    else:
        # Batch of images case
        images_tensor = torch.from_numpy(images_int).permute(0, 3, 1, 2)
    clip_score = clip_score_fn(images_tensor, prompts).detach()
    return round(float(clip_score), 4)

repo_id = "stablediffusionapi/interiordesignsuperm"
pipeline = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipeline.to("cuda")
#gc.collect()
#torch.cuda.empty_cache()

batch_size = 1
num_prompts = len(prompts)
total_images = num_prompts * batch_size
num_batches = total_images // batch_size

clip_scores = []
for i in range(num_batches):
    batch_prompts = prompts[i * batch_size:(i + 1) * batch_size]
    print(batch_prompts)
    if not all(batch_prompts):  # Check if any prompt is empty
        continue
    images = pipeline(prompt=batch_prompts, guidance_scale=3.5, num_inference_steps=50, num_images_per_prompt=10, output_type="np").images
    #gc.collect()
    #torch.cuda.empty_cache()
    for j, prompt in enumerate(batch_prompts):
        prompt_images = images[j * 10:(j + 1) * 10]  # Get the batch of images for the current prompt
        prompt_clip_scores = [calculate_clip_score(image, [prompt]) for image in prompt_images]
        prompt_clip_score = max(prompt_clip_scores)  # Use the maximum score or np.mean(prompt_clip_scores) for the mean
        clip_scores.append(prompt_clip_score)
    gc.collect()
    torch.cuda.empty_cache()

nofreeu_clip_score = np.mean(clip_scores)
print(f"No FreeU CLIP score: {nofreeu_clip_score}")

repo_id = "stablediffusionapi/interiordesignsuperm"
pipeline = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipeline.to("cuda")

batch_size = 1
num_prompts = len(prompts)
total_images = num_prompts * batch_size
num_batches = total_images // batch_size

clip_scores = []
for i in range(num_batches):
    batch_prompts = prompts[i * batch_size:(i + 1) * batch_size]
    print(batch_prompts)
    if not all(batch_prompts):  # Check if any prompt is empty
        continue
    register_free_upblock2d(pipeline, b1=1.3, b2=1.5, s1=0.9, s2=0.2)
    register_free_crossattn_upblock2d(pipeline, b1=1.3, b2=1.5, s1=0.9, s2=0.2)
    images = pipeline(prompt=batch_prompts, guidance_scale=3.5, num_inference_steps=50, num_images_per_prompt=10, output_type="np").images
    for j, prompt in enumerate(batch_prompts):
        image = images[j]
        clip_score = calculate_clip_score(image, prompt)
        clip_scores.append(clip_score)
    torch.cuda.empty_cache()
    gc.collect()

freeu_clip_score = np.mean(clip_scores)
print(f"FreeU CLIP score: {freeu_clip_score}")
