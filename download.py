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

tokenizer = CLIPTokenizerWithEmbeddings.from_pretrained(model_path, subfolder="tokenizer")

scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, tokenizer=tokenizer, safety_checker=None, torch_dtype=torch.float16, use_auth_token=HF_AUTH_TOKEN)


pipe.tokenizer = CLIPTokenizerWithEmbeddings.from_pretrained(model_path, subfolder="tokenizer")
pipe.tokenizer.load_embedding('<flora-marble>', './FloralMarble-400.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<flora-marble250>', './FloralMarble-250.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<flora-marble150>', './FloralMarble-150.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<photo-helper>', './PhotoHelper.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<lysergian-dreams>', './LysergianDreams-3600.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<urban-jungle>', './UrbanJungle.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<cinema-helper>', './CinemaHelper.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<neg-mutation>', './NegMutation-2400.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<car-helper>', './CarHelper.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<hyper-fluid>', './HyperFluid.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<double-exposure>', './dblx.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<pencil-graphite>', './ppgra.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<viking-punk>', './VikingPunk.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<gigachad>', './GigaChad.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<glass-case>', './kc16-v4-5000.pt', pipe.text_encoder)
pipe.tokenizer.load_embedding('<action-helper>', './ActionHelper.pt', pipe.text_encoder)

