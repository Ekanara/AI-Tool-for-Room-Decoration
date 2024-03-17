""
from src.demo.download import check_and_download, download_all
# check_and_download()
#download_all()

from src.demo.demo import create_demo_move, create_demo_appearance, create_demo_drag, create_demo_face_drag, create_demo_paste,create_demo_generate
from src.demo.model import InteriorModels

#import cv2
import gradio as gr

# main demo
pretrained_model_path = "runwayml/stable-diffusion-v1-5"
model = InteriorModels(pretrained_model_path=pretrained_model_path)

DESCRIPTION = '''
<div style="text-align: center; font-size: 30px;">
    <p>😭😭<a href="https://github.com/Ekanara/AI-Tool-for-Room-Decoration">AI-Tool-for-Room-Decoration</a>😭😭</p>
</div>
'''
#with gr.Blocks(css=".gradio-container {background: url('file=background.png'); background-size: cover}") as demo:
with gr.Blocks(css="css.style") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('Generate Image'):
          create_demo_generate(model.run_generate_style)
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