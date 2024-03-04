"""
from src.demo.download import check_and_download, download_all
# check_and_download()
#download_all()

from src.demo.demo import create_demo_move, create_demo_appearance, create_demo_drag, create_demo_face_drag, create_demo_paste
from src.demo.model import InteriorModels

#import cv2
import gradio as gr

# main demo
pretrained_model_path = "runwayml/stable-diffusion-v1-5"
model = InteriorModels(pretrained_model_path=pretrained_model_path)
"""
DESCRIPTION = '# 🐉🐉[DragonDiffusion V1.0](https://github.com/MC-E/DragonDiffusion)🐉🐉'

DESCRIPTION += f'<p>Gradio demo for [DragonDiffusion](https://arxiv.org/abs/2307.02421) and [DiffEditor](https://arxiv.org/abs/2307.02421). If it is helpful, please help to recommend [[GitHub Repo]](https://github.com/MC-E/DragonDiffusion) to your friends 😊 </p>'
"""
with gr.Blocks(css='style.css') as demo:
  #  gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('Appearance Modulation'):
            create_demo_appearance(model.run_appearance)
        with gr.TabItem('Object Moving & Resizing'):
            create_demo_move(model.run_move)
        #with gr.TabItem('Face Modulation'):
        #   create_demo_face_drag(model.run_drag_face)
        with gr.TabItem('Content Dragging'):
            create_demo_drag(model.run_drag)
        with gr.TabItem('Object Pasting'):
            create_demo_paste(model.run_paste)

demo.queue(concurrency_count=3, max_size=20)
demo.launch(server_name="0.0.0.0", share=True)"""
from diffusers import DiffusionPipeline

import torch
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline
from cog_sdxl.dataset_and_utils import TokenEmbeddingsHandler
from diffusers.models import AutoencoderKL


pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
).to("cuda")

pipe.load_lora_weights("fofr/sdxl-tng-interior", weight_name="lora.safetensors")

text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

embedding_path = hf_hub_download(repo_id="fofr/sdxl-tng-interior", filename="embeddings.pti", repo_type="model")
embhandler = TokenEmbeddingsHandler(text_encoders, tokenizers)
embhandler.load_embeddings(embedding_path)
prompt="A photo in the style of <s0><s1>, interior, house - sustainable, minimalist, organic, light-filled, dynamic, efficient, autonomous, connected, harmonious, innovative, detailed, 8k, high resolution, sharp focus"
images = pipe(
    prompt,
    cross_attention_kwargs={"scale": 0.8},
).images
#your output image
images[0]