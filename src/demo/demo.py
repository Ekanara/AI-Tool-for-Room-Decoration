import gradio as gr
import numpy as np
from PIL import Image
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
from src.demo.utils import get_point, store_img, get_point_move, store_img_move, clear_points, upload_image_move, segment_with_points, segment_with_points_paste, fun_clear, paste_with_mask_and_offset

def calculate_ssim(original_image, modified_image):
    print(f"original_image shape: {original_image.shape}")
    # If modified_image is a list, assume it's a list of images
    # and calculate the SSIM for each image
    if not isinstance(modified_image, list):
        return _extracted_from_calculate_ssim_18(modified_image, original_image)
    ssim_values = []
    for image in modified_image:
        print(f"modified_image shape: {image.shape}")
        original_tensor = torch.from_numpy(np.array(original_image)).permute(2, 0, 1).unsqueeze(0).float()
        modified_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float()
        modified_tensor = F.interpolate(modified_tensor, size=original_tensor.shape[-2:], mode='bilinear', align_corners=True)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        ssim_value = ssim(original_tensor, modified_tensor)
        ssim_values.append(f"{ssim_value:.4f}")
    return ", ".join(ssim_values)


# TODO Rename this here and in `calculate_ssim`
def _extracted_from_calculate_ssim_18(modified_image, original_image):
    # If modified_image is a single image
    print(f"modified_image shape: {modified_image.shape}")
    original_tensor = torch.from_numpy(np.array(original_image)).permute(2, 0, 1).unsqueeze(0).float()
    modified_tensor = torch.from_numpy(np.array(modified_image)).permute(2, 0, 1).unsqueeze(0).float()
    modified_tensor = F.interpolate(modified_tensor, size=original_tensor.shape[-2:], mode='bilinear', align_corners=True)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_value = ssim(original_tensor, modified_tensor)
    return f"{ssim_value:.4f}"

def create_demo_generate(runner):
    DESCRIPTION = """
        ## Image Generation
        Usage:
        - Choose a color tone, style, and room.
        - Adjust the advanced options if needed.
        - Click the `Generate` button to generate the image.
        """
    with gr.Blocks() as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        max_lines=1,
                        placeholder="Enter your prompt",
                        container=False,
                    )
                with gr.Row():
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        max_lines=1,
                        placeholder="Enter your negative prompt",
                        container=False,
                )
                with gr.Box():
                    guidance_scale = gr.Slider(label="Classifier guidance strength", value=3.5, minimum=0, maximum=10,
                                                step=0.1)
                    """
                    energy_scale = gr.Slider(label="Classifier guidance strength (x1e3)", value=0.5, minimum=0, maximum=10,
                                            step=0.1)
                    """
                    height = gr.Slider(label="Height", value=720, minimum=428, maximum=1024, step=8)

                    width = gr.Slider(label="width", value=1024, minimum=428, maximum=960, step=8)
                    with gr.Accordion('Advanced options', open=False):
                        seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                        b1 = gr.Slider(
                            label="Backbone 1 (b1: detail 1)",
                            minimum=0.8,
                            maximum=1.8,
                            step=0.1,
                            value=1.3,
                            interactive=True)
                        b2 = gr.Slider(
                            label="Backbone 2 (b2: detail 2)",
                            minimum=1,
                            maximum=2,
                            step=0.1,
                            value=1.4,
                            interactive=True)
                        s1 = gr.Slider(
                            label="Skip connection 1 (s1: contrast)",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.5,
                            interactive=True)
                        s2 = gr.Slider(
                            label="Skip connection 2 (s2: contrast)",
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0.2,
                            interactive=True)

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    output = gr.Image(type="pil")
                    with gr.Row():
                        run_button = gr.Button("Generate")

        prompt.submit(fn=runner, inputs=[prompt, negative_prompt, guidance_scale, height, width, b1, b2, s1, s2], outputs=[output])
        run_button.click(fn=runner, inputs=[prompt, negative_prompt, guidance_scale, height, width, b1, b2, s1, s2], outputs=[output])
    return demo

def create_demo_generate_nofreeu(runner):
    DESCRIPTION = """
        ## Image Generation
        Usage:
        - Choose a color tone, style, and room.
        - Adjust the advanced options if needed.
        - Click the `Generate` button to generate the image.
        """
    with gr.Blocks() as demo:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        max_lines=1,
                        placeholder="Enter your prompt",
                        container=False,
                    )
                with gr.Row():
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        max_lines=1,
                        placeholder="Enter your negative prompt",
                        container=False,
                )
                with gr.Box():
                    guidance_scale = gr.Slider(label="Classifier guidance strength", value=3.5, minimum=0, maximum=10,
                                                step=0.1)
                    """
                    energy_scale = gr.Slider(label="Classifier guidance strength (x1e3)", value=0.5, minimum=0, maximum=10,
                                            step=0.1)
                    """
                    height = gr.Slider(label="Height", value=720, minimum=428, maximum=1024, step=8)

                    width = gr.Slider(label="Width", value=1024, minimum=428, maximum=960, step=8)
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    output = gr.Image(type="pil")
                    with gr.Row():
                        run_button = gr.Button("Generate")

        prompt.submit(fn=runner, inputs=[prompt, negative_prompt, guidance_scale, height, width], outputs=[output])
        run_button.click(fn=runner, inputs=[prompt, negative_prompt, guidance_scale, height, width], outputs=[output])
        # Define the custom CSS style for the "Generate" button
    return demo

def create_demo_move(runner):
    DESCRIPTION = """
    ## Object Moving & Resizing
    Usage:
    - Upload a source image, and then draw a box to generate the mask corresponding to the editing object.
    - Label the object's movement path on the source image.
    - Label reference region. (optional)
    - Add a text description to the image and click the `Edit` button to start editing."""

    with gr.Blocks() as demo:
        original_image = gr.State(value=None) # store original image
        mask_ref = gr.State(value=None)
        selected_points = gr.State([])
        global_points = gr.State([])
        global_point_label = gr.State([])
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## 1. Draw box to mask target object")
                    img_draw_box = gr.Image(source='upload', label="Draw box", interactive=True, type="numpy")

                    gr.Markdown("## 2. Draw arrow to describe the movement")
                    img = gr.Image(source='upload', label="Original image", interactive=True, type="numpy")

                    gr.Markdown("## 3. Label reference region (Optional)")
                    img_ref = gr.Image(tool="sketch", label="Original image", interactive=True, type="numpy")

                    gr.Markdown("## 4. Prompt")
                    prompt = gr.Textbox(label="Prompt")

                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

                    with gr.Box():
                        guidance_scale = gr.Slider(label="Classifier-free guidance strength", value=4, minimum=1, maximum=10, step=0.1)
                        energy_scale = gr.Slider(label="Classifier guidance strength (x1e3)", value=0.5, minimum=0, maximum=10, step=0.1)
                        max_resolution = gr.Slider(label="Resolution", value=768, minimum=428, maximum=1024, step=1)
                        with gr.Accordion('Advanced options', open=False):
                            seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                            resize_scale = gr.Slider(
                                        label="Object resizing scale",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=1,
                                        interactive=True)
                            w_edit = gr.Slider(
                                        label="Weight of moving strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=4,
                                        interactive=True)
                            w_content = gr.Slider(
                                        label="Weight of content consistency strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=6,
                                        interactive=True)
                            w_contrast = gr.Slider(
                                        label="Weight of contrast strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=0.2,
                                        interactive=True)
                            w_inpaint = gr.Slider(
                                        label="Weight of inpainting strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=0.8,
                                        interactive=True)
                            SDE_strength = gr.Slider(
                                        label="Flexibility strength",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.4,
                                        interactive=True)
                            ip_scale = gr.Slider(
                                        label="Image prompt scale",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.1,
                                        interactive=True)
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    mask = gr.Image(source='upload', label="Mask of object", interactive=True, type="numpy")
                    im_w_mask_ref = gr.Image(label="Mask of reference region", interactive=True, type="numpy")

                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery(columns=1, height='auto')

            img.select(
                get_point_move,
                [original_image, img, selected_points],
                [img, original_image, selected_points],
            )
            img_draw_box.select(
                segment_with_points,
                inputs=[img_draw_box, original_image, global_points, global_point_label, img],
                outputs=[img_draw_box, original_image, mask, global_points, global_point_label, img, img_ref]
            )
            img_ref.edit(
                store_img_move,
                [img_ref],
                [original_image, im_w_mask_ref, mask_ref]
            )
            ssim_score = gr.Textbox(label="SSIM Score")
        def on_run_button_click(original_image, *args):
            output = runner(original_image, *args)
            ssim_value = calculate_ssim(original_image, output)
            return output, ssim_value
            
        run_button.click(
            fn=on_run_button_click,
            inputs=[original_image, mask, mask_ref, prompt, resize_scale, w_edit, w_content, w_contrast, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale],
            outputs=[output, ssim_score]
        )
        clear_button.click(fn=fun_clear, inputs=[original_image, global_points, global_point_label, selected_points, mask_ref, mask, img_draw_box, img, im_w_mask_ref], outputs=[original_image, global_points, global_point_label, selected_points, mask_ref, mask, img_draw_box, img, im_w_mask_ref])
    return demo

def create_demo_appearance(runner):
    DESCRIPTION = """
    ## Appearance Modulation
    Usage:
    - Upload a source image, and an appearance reference image.
    - Label object masks on these two image.
    - Add a text description to the image and click the `Edit` button to start editing."""

    with gr.Blocks() as demo:
        original_image_base = gr.State(value=None)
        global_points_base = gr.State([])
        global_point_label_base = gr.State([])
        global_points_replace = gr.State([])
        global_point_label_replace = gr.State([])
        original_image_replace = gr.State(value=None)
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## 1. Upload image & Draw box to generate mask")
                    img_base = gr.Image(source='upload', label="Original image", interactive=True, type="numpy")
                    img_replace = gr.Image(source="upload", label="Reference image", interactive=True, type="numpy")

                    gr.Markdown("## 2. Prompt")
                    prompt = gr.Textbox(label="Prompt of original image")
                    prompt_replace = gr.Textbox(label="Prompt of reference image")

                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

                    with gr.Box():
                        guidance_scale = gr.Slider(label="Classifier-free guidance strength", value=5, minimum=1, maximum=10, step=0.1)
                        energy_scale = gr.Slider(label="Classifier guidance strength (x1e3)", value=2, minimum=0, maximum=10, step=0.1)
                        max_resolution = gr.Slider(label="Resolution", value=768, minimum=428, maximum=1024, step=1)
                        with gr.Accordion('Advanced options', open=False):
                            seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                            w_edit = gr.Slider(
                                        label="Weight of moving strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=3.5,
                                        interactive=True)
                            w_content = gr.Slider(
                                        label="Weight of content consistency strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=5,
                                        interactive=True)
                            SDE_strength = gr.Slider(
                                        label="Flexibility strength",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.4,
                                        interactive=True)
                            ip_scale = gr.Slider(
                                        label="Image prompt scale",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.1,
                                        interactive=True)

            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    with gr.Row():
                        mask_base = gr.Image(source='upload', label="Mask of editing object", interactive=False, type="numpy")
                        mask_replace = gr.Image(tool="upload", label="Mask of reference object", interactive=False, type="numpy")
                    
                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery(columns=1, height='auto')
                    ssim_score = gr.Textbox(label="SSIM Score")

        img_base.select(
            segment_with_points, 
            inputs=[img_base, original_image_base, global_points_base, global_point_label_base], 
            outputs=[img_base, original_image_base, mask_base, global_points_base, global_point_label_base]
        )
        img_replace.select(
            segment_with_points, 
            inputs=[img_replace, original_image_replace, global_points_replace, global_point_label_replace], 
            outputs=[img_replace, original_image_replace, mask_replace, global_points_replace, global_point_label_replace]
        )

        def on_run_button_click(original_image_base, *args):
            output = runner(original_image_base, *args)
            ssim_value = calculate_ssim(original_image_base, output)
            return output, ssim_value
            
        clear_button.click(fn=fun_clear, inputs=[original_image_base, original_image_replace, global_points_base, global_points_replace, global_point_label_base, global_point_label_replace, img_base, img_replace, mask_base, mask_replace], outputs=[original_image_base, original_image_replace, global_points_base, global_points_replace, global_point_label_base, global_point_label_replace, img_base, img_replace, mask_base, mask_replace])

        run_button.click(fn=on_run_button_click, inputs=[original_image_base, mask_base, original_image_replace, mask_replace, prompt, prompt_replace, w_edit, w_content, seed, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale], outputs=[output, ssim_score])
    return demo

def create_demo_drag(runner):
    DESCRIPTION = """
    ## Content Dragging
    Usage:
    - Upload a source image.
    - Draw a mask on the source image.
    - Label the content's movement path on the masked image.
    - Add a text description to the image and click the `Edit` button to start editing."""

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(DESCRIPTION)
        original_image = gr.State(value=None) # store original image
        mask = gr.State(value=None) # store mask
        selected_points = gr.State([])
        with gr.Column(scale=2):
            with gr.Row():
                img_m = gr.Image(source='upload', tool="sketch", label="Original Image", interactive=True, type="numpy")
                img = gr.Image(source='upload', label="Original Image", interactive=True, type="numpy")
            img.select(
                get_point,
                [img, selected_points],
                [img],
            )
            img_m.edit(
                store_img,
                [img_m],
                [original_image, img, mask]
            )
            with gr.Row():
                run_button = gr.Button("Edit")
                clear_button = gr.Button("Clear points")
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(label="Prompt")
                    with gr.Box():
                        seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                        guidance_scale = gr.Slider(label="Classifier-free guidance strength", value=4, minimum=1, maximum=10, step=0.1)
                        energy_scale = gr.Slider(label="Classifier guidance strength (x1e3)", value=2, minimum=0, maximum=10, step=0.1)
                        max_resolution = gr.Slider(label="Resolution", value=768, minimum=428, maximum=1024, step=1)
                        with gr.Accordion('Advanced options', open=False):
                            w_edit = gr.Slider(
                                        label="Weight of moving strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=4,
                                        interactive=True)
                            w_content = gr.Slider(
                                        label="Weight of content consistency strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=6,
                                        interactive=True)
                            w_inpaint = gr.Slider(
                                        label="Weight of inpainting strength",
                                        minimum=0,
                                        maximum=10,
                                        step=0.1,
                                        value=0.2,
                                        interactive=True)
                            SDE_strength = gr.Slider(
                                        label="Flexibility strength",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.4,
                                        interactive=True)
                            ip_scale = gr.Slider(
                                        label="Image prompt scale",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.1,
                                        interactive=True)
                with gr.Column(scale=2):
                    gr.Markdown("<h5><center>Results</center></h5>")
                    output = gr.Gallery(columns=1, height='auto')
                    ssim_score = gr.Textbox(label="SSIM Score")

        def on_run_button_click(original_image, *args):
            output = runner(original_image, *args)
            ssim_value = calculate_ssim(original_image, output)
            return output, ssim_value
        run_button.click(
            fn=on_run_button_click,
            inputs=[original_image, mask, prompt, w_edit, w_content, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale], 
            outputs=[output, ssim_score]
        )
        clear_button.click(fn=clear_points, inputs=[img_m], outputs=[selected_points, img])
    return demo

def create_demo_paste(runner):
    DESCRIPTION = """
    ## Object Pasting
    Usage:
    - Upload a reference image, having the target object.
    - Label object masks on the reference image.
    - Upload a background image.
    - Modulate the size and position of the object after pasting.
    - Add a text description to the image and click the `Edit` button to start editing."""

    with gr.Blocks() as demo:
        global_points = gr.State([])
        global_point_label = gr.State([])
        original_image = gr.State(value=None) 
        mask_base = gr.State(value=None) 

        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# INPUT")
                    gr.Markdown("## 1. Upload image & Draw box to generate mask")
                    img_base = gr.Image(source='upload', label="Original image", interactive=True, type="numpy")
                    img_replace = gr.Image(source="upload", label="Reference image", interactive=True, type="numpy") 
                    gr.Markdown("## 2. Paste position & size")
                    dx = gr.Slider(
                        label="Horizontal movement",
                        minimum=-1000,
                        maximum=1000,
                        step=1,
                        value=0,
                        interactive=True
                    )
                    dy = gr.Slider(
                        label="Vertical movement",
                        minimum=-1000,
                        maximum=1000,
                        step=1,
                        value=0,
                        interactive=True
                    )
                    resize_scale = gr.Slider(
                        label="Resize object",
                        minimum=0,
                        maximum=1.5,
                        step=0.1,
                        value=1,
                        interactive=True
                    )
                    gr.Markdown("## 3. Prompt")
                    prompt = gr.Textbox(label="Prompt")
                    prompt_replace = gr.Textbox(label="Prompt of reference image")
                    with gr.Row():
                        run_button = gr.Button("Edit")
                        clear_button = gr.Button("Clear")

                    with gr.Box():
                        guidance_scale = gr.Slider(label="Classifier-free guidance strength", value=4, minimum=1, maximum=10, step=0.1)
                        energy_scale = gr.Slider(label="Classifier guidance strength (x1e3)", value=1.5, minimum=0, maximum=10, step=0.1)
                        max_resolution = gr.Slider(label="Resolution", value=768, minimum=428, maximum=1024, step=1)
                        with gr.Accordion('Advanced options', open=False):
                            seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                            w_edit = gr.Slider(
                                        label="Weight of pasting strength",
                                        minimum=0,
                                        maximum=20,
                                        step=0.1,
                                        value=4,
                                        interactive=True)
                            w_content = gr.Slider(
                                        label="Weight of content consistency strength",
                                        minimum=0,
                                        maximum=20,
                                        step=0.1,
                                        value=6,
                                        interactive=True)
                            SDE_strength = gr.Slider(
                                        label="Flexibility strength",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.4,
                                        interactive=True)
                            ip_scale = gr.Slider(
                                        label="Image prompt scale",
                                        minimum=0,
                                        maximum=1,
                                        step=0.1,
                                        value=0.1,
                                        interactive=True)

            with gr.Column():
                    with gr.Box():
                        gr.Markdown("# OUTPUT")
                        mask_base_show = gr.Image(source='upload', label="Mask of editing object", interactive=True, type="numpy")
                        gr.Markdown("<h5><center>Results</center></h5>")
                        output = gr.Gallery(columns=1, height='auto')
                        ssim_score = gr.Textbox(label="SSIM Score")

        img_replace.select(
            segment_with_points_paste, 
            inputs=[img_replace, original_image, global_points, global_point_label, img_base, dx, dy, resize_scale], 
            outputs=[img_replace, original_image, mask_base_show, global_points, global_point_label, mask_base]
        )
        img_replace.edit(
            upload_image_move,
            inputs = [img_replace, original_image],
            outputs = [original_image]
        )
        dx.change(
            paste_with_mask_and_offset,
            inputs = [img_replace, img_base, mask_base, dx, dy, resize_scale],
            outputs =  mask_base_show
        )
        dy.change(
            paste_with_mask_and_offset,
            inputs = [img_replace, img_base, mask_base, dx, dy, resize_scale],
            outputs =  mask_base_show
        )
        resize_scale.change(
            paste_with_mask_and_offset,
            inputs = [img_replace, img_base, mask_base, dx, dy, resize_scale],
            outputs =  mask_base_show
        )
        def on_run_button_click(img_base, *args):
            output = runner(img_base, *args)
            ssim_value = calculate_ssim(img_base, output)
            return output, ssim_value

        clear_button.click(fn=fun_clear, inputs=[original_image, global_points, global_point_label, img_replace, mask_base, img_base], outputs=[original_image, global_points, global_point_label, img_replace, mask_base, img_base])
        run_button.click(
            fn=on_run_button_click,
            inputs=[img_base, mask_base, original_image, prompt, prompt_replace, w_edit, w_content, seed, guidance_scale, energy_scale, dx, dy, resize_scale, max_resolution, SDE_strength, ip_scale],
            outputs=[output, ssim_score]
        )
    return demo

"""
def create_demo_generate(runner):
    DESCRIPTION = 
        ## Image Generation
        Usage:
        - Choose a color tone, style, and room.
        - Adjust the advanced options if needed.
        - Click the `Generate` button to generate the image.
        
    with gr.Blocks() as demo:
        color_tones = ['warm', 'cool', 'dark', 'light']  # Option List
        styles = ['wooden', 'modern', 'vintage', 'minimalist']  # Option List
        rooms = ['bedroom', 'living room', 'kitchen', 'bathroom']  # Option List
        prompt = gr.State("")
        def update_prompt():
            prompt = gr.State(
                f"Generate a {color_tone_dropdown.value} {style_dropdown.value} {room_dropdown.value}")
            return prompt
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            with gr.Column():

                with gr.Box():
                    gr.Markdown("Choose a color tone")
                    color_tone_dropdown = gr.Dropdown(color_tones)
                with gr.Box():
                    gr.Markdown("Choose a style")
                    style_dropdown = gr.Dropdown(styles)
                with gr.Box():
                    gr.Markdown("Choose a room")
                    room_dropdown = gr.Dropdown(rooms)

                with gr.Box():
                    guidance_scale = gr.Slider(label="Classifier-free guidance strength", value=4, minimum=1, maximum=10,
                                               step=0.1)
                    energy_scale = gr.Slider(label="Classifier guidance strength (x1e3)", value=0.5, minimum=0, maximum=10,
                                              step=0.1)
                    max_resolution = gr.Slider(label="Resolution", value=768, minimum=428, maximum=1024, step=1)
                    #with gr.Accordion('Advanced options', open=False):
                    seed = gr.Slider(label="Seed", value=42, minimum=0, maximum=10000, step=1, randomize=False)
                    SDE_strength = gr.Slider(
                        label="Flexibility strength",
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0.4,
                        interactive=True)
                    ip_scale = gr.Slider(
                        label="Image prompt scale",
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        value=0.1,
                        interactive=True)
            with gr.Column():
                with gr.Box():
                    gr.Markdown("# OUTPUT")
                    output = gr.outputs.Image(type="pil")
                    with gr.Row():
                        run_button = gr.Button("Generate")
                        if run_button.click:
                            color_tone_dropdown.change(fn=lambda: update_prompt())
                            style_dropdown.change(fn=lambda: update_prompt())
                            room_dropdown.change(fn=lambda: update_prompt())
                            print(prompt)

            run_button.click(fn=runner,inputs=[prompt, guidance_scale, max_resolution], outputs=[output])
   
    return demo
"""
