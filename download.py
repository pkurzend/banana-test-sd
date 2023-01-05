# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import os

from transformers import CLIPTextModel, CLIPTokenizer
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DPMSolverMultistepScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from CLIPTokenizerWithEmbeddings import CLIPTokenizerWithEmbeddings

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")


model_path = 'stabilityai/stable-diffusion-2-1'


scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
tokenizer = CLIPTokenizerWithEmbeddings.from_pretrained(model_path, subfolder="tokenizer")
pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, tokenizer=tokenizer, safety_checker=None, torch_dtype=torch.float16, use_auth_token=HF_AUTH_TOKEN)




