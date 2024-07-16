from semantic_aug.augmentations.textual_inversion import TextualInversion
from diffusers import StableDiffusionPipeline
from itertools import product
from torch import autocast
from PIL import Image

from tqdm import trange
import os
import torch
import argparse
import numpy as np
import random


DEFAULT_ERASURE_CKPT = (
    "/projects/rsalakhugroup/btrabucc/esd-models/" + 
    "compvis-word_airplane-method_full-sg_3-ng_1-iter_1000-lr_1e-05/" + 
    "diffusers-word_airplane-method_full-sg_3-ng_1-iter_1000-lr_1e-05.pt")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Stable Diffusion inference script")

    parser.add_argument("--model_ckpt", type=str, required=True)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generate", type=int, default=50)

    parser.add_argument("--prompt", type=str, default="a photo of a <patio>")
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--erasure-ckpt-name", type=str, default=DEFAULT_ERASURE_CKPT)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    const_out = args.out

    for dir_ckpt in os.listdir(args.model_ckpt):

        for clas in os.listdir(os.path.join(args.model_ckpt, dir_ckpt)):

            args.prompt = "a photo of a " + "<" + clas + ">"
            args.out = os.path.join(const_out, clas) + "/"

            if os.path.isdir(args.out):
                pass
            else:
                os.mkdir(args.out)

            args.embed_path = args.model_ckpt + dir_ckpt + "/" + clas + "/learned_embeds.bin"

            pipe = StableDiffusionPipeline.from_pretrained(
                args.model_path, use_auth_token=True,
                revision="fp16", 
                torch_dtype=torch.float16
            ).to('cuda')

            aug = TextualInversion(args.embed_path, model_path=args.model_path)
            pipe.tokenizer = aug.pipe.tokenizer
            pipe.text_encoder = aug.pipe.text_encoder

            pipe.set_progress_bar_config(disable=True)
            pipe.safety_checker = None

            ckpt_num = dir_ckpt.split("-")[1]
            for idx in trange(100, 
                            desc="Generating Images"):

                seed = random.randint(0,2000)
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                with autocast('cuda'):

                    image = pipe(
                        args.prompt, 
                        guidance_scale=args.guidance_scale
                    ).images[0]

                image.save(os.path.join(args.out, f"{idx}_{ckpt_num}.png"))