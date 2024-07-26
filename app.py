import gradio as gr
import spaces
import os
import sys
import subprocess
import numpy as np
from PIL import Image
import cv2

import torch

from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image

# load pipeline
controlnet_canny = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny")
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    controlnet=controlnet_canny
).to("cuda", torch.float16)

def resize_image(input_path, output_path, target_height):
    # Open the input image
    img = Image.open(input_path)

    # Calculate the aspect ratio of the original image
    original_width, original_height = img.size
    original_aspect_ratio = original_width / original_height

    # Calculate the new width while maintaining the aspect ratio and the target height
    new_width = int(target_height * original_aspect_ratio)

    # Resize the image while maintaining the aspect ratio and fixing the height
    img = img.resize((new_width, target_height), Image.LANCZOS)

    # Save the resized image
    img.save(output_path)

    return output_path, new_width, target_height


@spaces.GPU(duration=90)
def infer(image_in, prompt, inference_steps, guidance_scale, control_weight, progress=gr.Progress(track_tqdm=True)):

    n_prompt = 'NSFW, nude, naked, porn, ugly'

    # Canny preprocessing
    image_to_canny = load_image(image_in)
    image_to_canny = np.array(image_to_canny)
    image_to_canny = cv2.Canny(image_to_canny, 100, 200)
    image_to_canny = image_to_canny[:, :, None]
    image_to_canny = np.concatenate([image_to_canny, image_to_canny, image_to_canny], axis=2)
    image_to_canny = Image.fromarray(image_to_canny)

    control_image = image_to_canny
 
    # infer
    image = pipe(
        prompt=prompt,
        negative_prompt=n_prompt,
        control_image=control_image, 
        controlnet_conditioning_scale=control_weight,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    
    image_redim, w, h = resize_image(image_in, "resized_input.jpg", 1024)
    image = image.resize((w, h), Image.LANCZOS)
    
    return image, gr.update(value=image_to_canny, visible=True)
   

css="""
#col-container{
    margin: 0 auto;
    max-width: 1080px;
}
"""
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("""
        # SD3 ControlNet

        Experiment with Stable Diffusion 3 ControlNet models proposed and maintained by the InstantX team.<br />
        Model card: [InstantX/SD3-Controlnet-Canny](https://huggingface.co/InstantX/SD3-Controlnet-Canny)
        """)
        
        with gr.Column():
            
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(label="Image reference", sources=["upload"], type="filepath")
                    prompt = gr.Textbox(label="Prompt")
                    
                    with gr.Accordion("Advanced settings", open=False):
                        with gr.Column():
                            with gr.Row():
                                inference_steps = gr.Slider(label="Inference steps", minimum=1, maximum=50, step=1, value=25)
                                guidance_scale = gr.Slider(label="Guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=7.0)
                            control_weight = gr.Slider(label="Control Weight", minimum=0.0, maximum=1.0, step=0.01, value=0.7)
                    
                    submit_canny_btn = gr.Button("Submit")
                    
                with gr.Column():
                    result = gr.Image(label="Result")
                    canny_used = gr.Image(label="Preprocessed Canny", visible=False)
                    


    submit_canny_btn.click(
        fn = infer,
        inputs = [image_in, prompt, inference_steps, guidance_scale, control_weight],
        outputs = [result, canny_used],
        show_api=False
    )
demo.queue().launch()