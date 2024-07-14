import cv2
import sys
import argparse
import numpy as np
from datasets import load_from_disk
import torch
from pathlib import Path
from matplotlib import pyplot as plt
import os
from PIL import Image
import sys
import json
import pickle
from lama_inpaint import inpaint_img_with_lama
from imagenet_num_to_label import num_to_label
from transformers import AutoImageProcessor, ResNetForImageClassification
import pandas as pd
from tqdm.auto import tqdm
import torchvision
import torch.nn as nn
# Grounding DINO
uncle_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Grounded-Segment-Anything', 'GroundingDINO'))
sys.path.append(uncle_path)
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import fill_img_with_sd
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points
from segment_anything import build_sam, SamPredictor

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

def load_image(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)

def show_mask_grounded_sam(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def to_3d(grayscale_image):
    # Open the grayscale image using Pillow
    # grayscale_image = Image.open("path/to/your/grayscale_image.jpg")

    # Convert the grayscale image to a NumPy array
    grayscale_array = np.array(grayscale_image)

    # Create a 3D array with identical values for each channel
    rgb_array = np.stack((grayscale_array, grayscale_array, grayscale_array), axis=-1)

    # Create an RGB PIL Image from the 3D array
    rgb_image = Image.fromarray(rgb_array)

    return rgb_image

def is_grayscale(image):
    """
    Check if the given image is grayscale or color.
    """
    # Get the mode of the image
    mode = image.mode

    # Check if the mode indicates grayscale
    return mode == "L"

def setup_args(parser):
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Input json file",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help=""
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help=""
    )
    parser.add_argument(
        "--image_output_path", type=str, required=True, help=""
    )
    parser.add_argument(
        "--model_ckpt_path", type=str, required=True, help=""
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Specify seed for reproducibility.",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic algorithms for reproducibility.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str,
        default='./pretrained_models/big-lama',
        help="The path to the lama checkpoint.",
    )
    parser.add_argument("--det_prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])

    dataset = pd.read_json(args.input_file, lines=True)
    config_file = args.config
    grounded_checkpoint = args.grounded_checkpoint
    sam_checkpoint = args.sam_ckpt
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    output_dir = args.output_dir
    device = "cuda"

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(config_file, grounded_checkpoint, device=device)
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-224")
    model2 = torchvision.models.vit_l_32(pretrained=False)
    num_ftrs = model2.fc.in_features
    model2.fc = nn.Linear(num_ftrs, 2)
    model2.load_state_dict(torch.load(args.model_ckpt_path))
    model2.to("cuda")
    model2.eval()
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    if args.dataset == "imagenet":
        spur_dataset = load_from_disk(args.dataset_path)

    wrong_pred_dict = {"img_id": [], "img_filename": [], "caption": [], "class": [], "label": [], "foreground": [], "wp_foreground": [], "background": [], "wp_background": [], "alternate_background": [], "wp_alternate_background": [], "pred_alt_background": [], "pred_rmv_foreground": []}

    image_save_path = args.image_output_path

    fail_pred = 0
    for xx in tqdm(range(len(dataset)), desc="Executing algo", total=len(dataset), unit="rows"):

        act_image = Image.open(dataset['img_filename'][xx])

        ori_pixel_values = processor(act_image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad():
            ori_logits = model2(ori_pixel_values)
        ori_predicted_label = torch.argmax(ori_logits, dim=-1).cpu().tolist()[0]

        if ori_predicted_label != dataset['class'][xx]:
            print(f"Wrong prediction for idx : {xx}")
            fail_pred = fail_pred + 1
            continue

        wrong_pred_dict["img_id"].append(dataset["img_id"][xx])
        wrong_pred_dict["img_filename"].append(dataset["img_filename"][xx])
        wrong_pred_dict["caption"].append(dataset["caption"][xx])
        wrong_pred_dict["class"].append(dataset["class"][xx])
        wrong_pred_dict["label"].append(dataset["label"][xx])
        wrong_pred_dict["foreground"].append(dataset["foreground"][xx])
        wrong_pred_dict["background"].append(dataset["background"][xx])
        wrong_pred_dict["alternate_background"].append(dataset["alternate_background"][xx])

        failed_foreground = []
        failed_background = []
        failed_alt_background = []
        pred_alt_background = []
        pred_rmv_foreground = []

        if len(dataset["alternate_background"][xx]) > 0 and len(dataset["alternate_background"][xx]) == len(dataset["background"][xx]):
            for j in range(len(dataset["alternate_background"][xx])):
                images = pipe(f"turn the background of {dataset['background'][xx][j]} into {dataset['alternate_background'][xx][j]}.", image=act_image.convert('RGB')).images
                pixel_values = processor(images[0], return_tensors="pt").pixel_values.to("cuda")
                with torch.no_grad():
                    logits = model2(pixel_values)
                predicted_labels = torch.argmax(logits, dim=-1).cpu().tolist()[0] 
                pred_alt_background.append(predicted_labels)
                if predicted_labels != ori_predicted_label:
                    print('*********** inspix2pix ***********')
                    print('Actual label: ' + str(dataset['class'][xx]))
                    print(f'Predicted label: {predicted_labels}')
                    failed_alt_background.append(dataset['alternate_background'][xx][j])
                    failed_background.append(dataset['background'][xx][j])
                    images[0].save(image_save_path + str(xx) + "_bg_" + dataset['background'][xx][j].replace(" ","-") + ".jpg")

        wrong_pred_dict["wp_alternate_background"].append(failed_alt_background)
        wrong_pred_dict["wp_background"].append(failed_background)
        wrong_pred_dict["pred_alt_background"].append(pred_alt_background)

        for j in range(len(dataset["foreground"][xx])):
            try:
                image_pil = act_image.convert('RGB')
                image_og = load_image(image_pil)

                img = np.array(image_pil)

                out_dir = Path(os.path.join(args.output_dir, f'{xx}_{j}'))
                out_dir.mkdir(parents=True, exist_ok=True)

                det_prompt = dataset["foreground"][xx][j]

                if args.dataset == "imagenet":
                    correct_class = num_to_label(dataset['class'][xx])
                elif args.dataset == "waterbird":
                    correct_class = "water_bird" if dataset['label'][xx] == 0 else "land_bird"
                elif args.dataset == "waterbird":
                    correct_class = "small_dog" if dataset['label'][xx] == 0 else "big_dog"

                boxes_filt, pred_phrases = get_grounding_output(
                    model, image_og, det_prompt, box_threshold, text_threshold, device=device
                )

                image = img
                predictor.set_image(image)

                size = image_pil.size
                H, W = size[1], size[0]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]

                boxes_filt = boxes_filt.cpu()
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

                masks, _, _ = predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes.to(device),
                    multimask_output = False
                )

                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                for mask in masks:
                    show_mask_grounded_sam(mask.cpu().numpy(), plt.gca(), random_color=True)
                for box, label in zip(boxes_filt, pred_phrases):
                    show_box(box.numpy(), plt.gca(), label)

                plt.axis('off')

                masks = torch.sum(masks, dim=0)
                mask = masks[0][0].cpu().numpy()
                masks = masks.cpu().numpy().astype(np.uint8) * 255

                # dilate mask to avoid unmasked edge effect
                if args.dilate_kernel_size is not None:
                    masks = dilate_mask(masks, args.dilate_kernel_size)

                for idx, mask in enumerate(masks):
                    mask_p = out_dir / f"mask_{idx}.png"
                    img_points_p = out_dir / f"with_points.png"
                    img_mask_p = out_dir / f"with_{Path(mask_p).name}"

                # fill the masked image
                for idx, mask in enumerate(masks):
                    if args.seed is not None:
                        torch.manual_seed(args.seed)
                    img_inpainted = inpaint_img_with_lama(
                        img, mask, args.lama_config, args.lama_ckpt)
                    pixel_values = processor([img_inpainted], return_tensors="pt").pixel_values.to("cuda")
                    with torch.no_grad():
                        logits = model2(pixel_values)
                    predicted_labels = torch.argmax(logits, dim=-1).cpu().tolist()[0]
                    with open(out_dir / "info.txt", 'w') as f:
                        f.write(f'prompt: {det_prompt}\n')
                        f.write(f'Predicted label: {predicted_labels}')
                    if predicted_labels != dataset['class'][xx]:
                        print('************ Inpaint Anything **********')
                        print('Actual label: ' + str(dataset['label'][xx]))
                        print(f'Predicted label: {predicted_labels}')
                        failed_foreground.append(dataset["foreground"][xx][j])
                        pred_rmv_foreground.append(predicted_labels)
                        img_inpainted_p = (image_save_path + str(xx) + "_fg_" + dataset['foreground'][xx][j].replace(" ","-") + ".jpg")
                        save_array_to_img(img_inpainted, img_inpainted_p)
            except Exception as e:
                print(e)

        wrong_pred_dict["wp_foreground"].append(failed_foreground)
        wrong_pred_dict["pred_rmv_foreground"].append(pred_rmv_foreground)

    print(f"Total failed predictions : {fail_pred}")
    pd.DataFrame(wrong_pred_dict).to_json(f"{args.output_dir}/{args.dataset}_wrong_preds.jsonl", orient="records", lines=True)