import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import base64
from io import BytesIO
import os



import os

from transformers import CLIPTextModel, CLIPTokenizer
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, DDPMScheduler, UNet2DConditionModel, AutoencoderKL




# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    
    # this will substitute the default PNDM scheduler for K-LMS  
    #lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    #model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=lms, use_auth_token=HF_AUTH_TOKEN).to("cuda")
    
    model_path = 'stabilityai/stable-diffusion-2-1'

    tokenizer = CLIPTokenizerWithEmbeddings.from_pretrained(model_path, subfolder="tokenizer")

    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    model = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, tokenizer=tokenizer, safety_checker=None, torch_dtype=torch.float16, use_auth_token=HF_AUTH_TOKEN).to("cuda")

    model.tokenizer.load_embedding('<flora-marble>', './FloralMarble-400.pt', model.text_encoder)
    model.tokenizer.load_embedding('<flora-marble250>', './FloralMarble-250.pt', model.text_encoder)
    model.tokenizer.load_embedding('<flora-marble150>', './FloralMarble-150.pt', model.text_encoder)
    model.tokenizer.load_embedding('<photo-helper>', './PhotoHelper.pt', model.text_encoder)
    model.tokenizer.load_embedding('<lysergian-dreams>', './LysergianDreams-3600.pt', model.text_encoder)
    model.tokenizer.load_embedding('<urban-jungle>', './UrbanJungle.pt', model.text_encoder)
    model.tokenizer.load_embedding('<cinema-helper>', './CinemaHelper.pt', model.text_encoder)
    model.tokenizer.load_embedding('<neg-mutation>', './NegMutation-2400.pt', model.text_encoder)
    model.tokenizer.load_embedding('<car-helper>', './CarHelper.pt', model.text_encoder)
    model.tokenizer.load_embedding('<hyper-fluid>', './HyperFluid.pt', model.text_encoder)
    model.tokenizer.load_embedding('<double-exposure>', './dblx.pt', model.text_encoder)
    model.tokenizer.load_embedding('<pencil-graphite>', './ppgra.pt', model.text_encoder)
    model.tokenizer.load_embedding('<viking-punk>', './VikingPunk.pt', model.text_encoder)
    model.tokenizer.load_embedding('<gigachad>', './GigaChad.pt', model.text_encoder)
    model.tokenizer.load_embedding('<glass-case>', './kc16-v4-5000.pt', model.text_encoder)
    model.tokenizer.load_embedding('<action-helper>', './ActionHelper.pt', model.text_encoder)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    negative_prompt = model_inputs.get('negative_prompt', '')
    height = model_inputs.get('height', 768)
    width = model_inputs.get('width', 768)
    num_inference_steps = model_inputs.get('num_inference_steps', 50)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    input_seed = model_inputs.get("seed",None)
    
    #If "seed" is not sent, we won't specify a seed in the call
    generator = None
    if input_seed != None:
        generator = torch.Generator("cuda").manual_seed(input_seed)
    
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    with torch.inference_mode():
        image = model(prompt,height=height, negative_prompt=negative_prompt, width=width,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,generator=generator)["sample"][0]
    
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
