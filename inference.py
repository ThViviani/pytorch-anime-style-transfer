import os
import torch
import argparse
import sys

from pathlib import Path
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from src.models.cycle_gan import *

HF_REPO_ID    = "ThViviani/cycle_gan_for_anime2human_style_transfer"  
CKPT_FILENAME = "epoch37-step191064.ckpt"         
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3), 
])
postprocess = transforms.Compose([
    transforms.Normalize((-1.,)*3, (2.,)*3), # (x_norm + 1) / 2
    transforms.Lambda(lambda t: t.clamp(0,1)),
    transforms.ToPILImage(),
])

def translate_image(input_path: str, output_path: str, model: L.LightningModule):
    img = Image.open(input_path).convert("RGB")
    x   = preprocess(img).unsqueeze(0).to(DEVICE)   # [1,3,256,256]
    with torch.no_grad():
        y = model.Gy(x)
    out_img = postprocess(y.squeeze(0).cpu())
    
    if os.path.isdir(output_path):
        base = os.path.basename(input_path)
        name, ext = os.path.splitext(base)
        if not ext:
            ext = ".png"
        output_file = os.path.join(output_path, f"{name}_translated{ext}")


    out_img.save(output_file)
    print(f"Saved translated image to {output_file}")

def get_last_checkpoint():
    local_ckpt = hf_hub_download(
        repo_id = HF_REPO_ID,
        filename= CKPT_FILENAME,
        repo_type="model",
    )

    model  = CycleGAN.load_from_checkpoint(local_ckpt,).to(DEVICE)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CycleGAN inference: translate human faces to anime style faces."
    )
    parser.add_argument("input",  help="path to input photo")
    
    parser.add_argument(
        "output", 
        help="where to save the result",
        nargs="?",
        default="."
    )
    
    args = parser.parse_args()
    if not os.path.isdir(args.output):
        sys.exit(
            f"Invalid output path: «{args.output}». "
            "Please make sure the directory exists and is writable."
        )

    translate_image(args.input, args.output, get_last_checkpoint())


