import transformers
transformers.set_seed(42)
from transformers import AutoProcessor, AutoModelForCausalLM
from datasets import load_from_disk
from tqdm.auto import tqdm
import pickle
from PIL import Image
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Process an input file and generate an output file.")

parser.add_argument(
    '-i', '--input_file_path',
    type=str,
    required=True,
    help="Path to the input file"
)

parser.add_argument(
    '-o', '--output_file_path',
    type=str,
    required=True,
    help="Path to the output file"
)

parser.add_argument(
    '-m', '--model',
    type=str,
    default="microsoft/git-large-coco",
    help="Model name or path."
)

args = parser.parse_args()

processor = AutoProcessor.from_pretrained(args.model, do_normalize=False)
model = AutoModelForCausalLM.from_pretrained(args.model)
model = model.to("cuda")

f = open(args.input_file_path, "r")
images = []
paths = []

for line in f:
    line = line.strip()
    paths.append(line)
    images.append(Image.open(line))

captions = []
pixel_values = processor(images=images, return_tensors="pt").pixel_values.to("cuda")
for i, pixel_value in tqdm(enumerate(pixel_values), total=len(images), unit="image", desc="Captioning Images"):
    generated_ids = model.generate(pixel_values=pixel_value.unsqueeze(0), max_length=100)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True, num_beams=5)[0]
    captions.append(generated_caption)

for image in images:
    image.close()

wb_dict = {"path": paths, "caption": captions}

df = pd.DataFrame.from_dict(wb_dict)

csv_file = args.output_file_path
df.to_csv(csv_file, index=False)