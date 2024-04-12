# from src.demo.download import check_and_download, download_all
# download_all()

from src.demo.demo import create_demo_move, create_demo_appearance, create_demo_drag, create_demo_paste, create_demo_generate, create_demo_generate_nofreeu
from src.demo.model import InteriorModels
from style_generate import run_generate_style
from style_generate_nofreeu import run_generate_style_nofreeu

import gradio as gr

pretrained_model_path = "runwayml/stable-diffusion-v1-5"
model = InteriorModels(pretrained_model_path=pretrained_model_path)

DESCRIPTION = '''
<div style="text-align: center; font-size: 30px;">
    <h1>😭😭<a href="https://github.com/Ekanara/AI-Tool-for-Room-Decoration">AI Tool for Room Decoration</a>😭😭</h1>
    <h2><a href="https://github.com/Ekanara/AI-Tool-for-Room-Decoration">Github</a></h2>
</div>
'''
#with gr.Blocks(css=".gradio-container {background: url('file=background.png'); background-size: cover}") as demo:
with gr.Blocks(css="css.style") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('Generate Image (No FreeU)'):
            create_demo_generate_nofreeu(run_generate_style_nofreeu)
        with gr.TabItem('Generate Image (With FreeU)'):
            create_demo_generate(run_generate_style)
        with gr.TabItem('Appearance Modulation'):
            create_demo_appearance(model.run_appearance)
        with gr.TabItem('Object Moving & Resizing'):
            create_demo_move(model.run_move)
        with gr.TabItem('Content Dragging'):
            create_demo_drag(model.run_drag)
        with gr.TabItem('Object Pasting'):
            create_demo_paste(model.run_paste)

demo.queue(concurrency_count=3, max_size=20)
demo.launch(server_name="127.0.0.1", share=True)