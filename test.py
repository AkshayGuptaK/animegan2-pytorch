import os
import argparse

from PIL import Image
import numpy as np

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator


torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_image(image_path, x32=False):
    img = Image.open(image_path).convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img

def select_checkpoint(model):
    if model == 'p':
        return 'paprika.pt'
    if model == 'f':
        return 'face_paint_512_v2.pt'
    if model == 'c':
        return 'celeba_distill.pt'
    return None

def process_image(net, image_path, output_path, device):
    print(f"processing {image_path}")
    if os.path.splitext(image_path)[-1].lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        return
            
    image = load_image(image_path, args.x32)

    with torch.no_grad():
        image = to_tensor(image).unsqueeze(0) * 2 - 1
        out = net(image.to(device), args.upsample_align).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        out = to_pil_image(out)

    out.save(output_path)
    print(f"image saved: {output_path}")

def load_checkpoint(net, checkpoint, device):
    net.load_state_dict(torch.load(f"./weights/{checkpoint}", map_location="cpu"))
    net.to(device).eval()
    print(f"model loaded: {checkpoint}")    

def do_conversion(net, device, model, dir, image_name):
    checkpoint = select_checkpoint(model)
    load_checkpoint(net, checkpoint, device)
    image_path = os.path.join(dir, image_name)
    [img, ext] = os.path.splitext(image_name)
    output_path = f"{args.dir}/{img}_{model}.{ext}"
    process_image(net, image_path, output_path, device)    

def test(args):
    device = args.device
    image_name = args.image
    model = args.model
    dir = args.dir
    os.makedirs(dir, exist_ok=True)
    
    net = Generator()

    if model == 'all':
        for mode in ['c', 'f', 'p']:
            do_conversion(net, device, mode, dir, image_name)
    else:    
        do_conversion(net, device, model, dir, image_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        default='all',
    )
    parser.add_argument(
        '--image', 
        type=str, 
        default='1.jpg',
    )
    parser.add_argument(
        '--dir', 
        type=str, 
        default='./samples/inputs',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
    )
    parser.add_argument(
        '--upsample_align',
        type=bool,
        default=False,
        help="Align corners in decoder upsampling layers"
    )
    parser.add_argument(
        '--x32',
        action="store_true",
        help="Resize images to multiple of 32"
    )
    args = parser.parse_args()
    
    test(args)
