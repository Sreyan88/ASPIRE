from datasets import load_from_disk
import os
from PIL import Image
from tqdm.auto import tqdm
import random
from itertools import combinations
random.seed(42)
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
nltk.download('punkt')  # Download the punkt tokenizer
nltk.download('wordnet')
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Process an input file and generate an output file.")
parser.add_argument(
    '--dataset',
    type=str,
    required=True
)
parser.add_argument(
    '--wrong_preds_jsonl_path',
    type=str,
    required=True
)

parser.add_argument(
    '--spurious_image_saved_path',
    type=str,
    required=True
)

parser.add_argument(
    '--non_spurious_images_save_path',
    type=str,
    required=True
)
args = parser.parse_args()

def are_phrases_same_object(phrase1, phrase2):
    vectorizer = CountVectorizer().fit_transform([phrase1, phrase2])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)[0][1]
    return cosine_sim >= 0.7  # Set a threshold for similarity

def find_similar_items(phrases):
    similar_items = []
    for phrase in phrases:
        is_similar = False
        for item_list in similar_items:
            if any(are_phrases_same_object(phrase[1], item[1]) for item in item_list):
                item_list.append(phrase)
                is_similar = True
                break
        if not is_similar:
            similar_items.append([phrase])
    return similar_items

def get_collapsed_categories(spur_objs):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    phrases = spur_objs

    phrases_lemmatized_stemmed = [(phrase, lemmatizer.lemmatize(stemmer.stem(phrase))) for phrase in phrases]

    similar_items_list = find_similar_items(phrases_lemmatized_stemmed)

    # Extract only the original phrases from the list of tuples
    result = [[phrase for phrase, _ in item_list] for item_list in similar_items_list]

    return result

spur_dataset = pd.read_json(args.wrong_preds_jsonl_path, lines=True)

directory_path = args.spurious_image_saved_path

base_img_path = args.non_spurious_images_save_path

# Get a list of all files in the given directory
file_names = os.listdir(directory_path)

# Filter only the image files (you can adjust the list of allowed extensions as needed)
allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif']

image_files = [file for file in file_names if os.path.splitext(file)[1].lower() in allowed_extensions]

df = pd.read_json(args.wrong_preds_jsonl_path, lines=True)

# Step 2: Create a new DataFrame by merging foreground and background lists
foreground_df = df.explode("foreground")[["class", "foreground"]]
foreground_df.columns = ["class", "object"]

background_df = df.explode("background")[["class", "background"]]
background_df.columns = ["class", "object"]

merged_df = pd.concat([foreground_df, background_df])
merged_df = merged_df.dropna()
merged_df["object"] = merged_df["object"].apply(lambda obj: obj.replace("-", " "))

# Step 3: Group by "class" and "object" and count occurrences
pivot_table = merged_df.groupby(["class", "object"]).size().unstack(fill_value=0)

labels_dict = {
    0: "landbird",
    1 : "waterbird"
}

class_top_k_dict = {}

obj_image_dict = {}

# Read and process each image
for idx, image_file in tqdm(enumerate(image_files)):
    name = image_file.split(".")[0]
    name_split = name.split("_")
    idx = int(name_split[0])
    type = "fg"
    obj = name_split[2].replace("-", " ")
    label = spur_dataset["class"][idx]

    if label not in obj_image_dict:
        obj_image_dict[label] = {obj: [image_file]}
    elif obj not in obj_image_dict[label]:
        obj_image_dict[label][obj] = [image_file]
    else:
        obj_image_dict[label][obj].append(image_file)


    if label not in class_top_k_dict:
        class_top_k_dict[label] = {"name": labels_dict[label], "fg": {}}

    if obj in class_top_k_dict[label][type]:
        class_top_k_dict[label][type][obj] = class_top_k_dict[label][type][obj] + 1
    else:
        class_top_k_dict[label][type][obj] = 1

image_count = 0
for key, value in obj_image_dict.items():
    for key1 in value:
        image_count = image_count + len(value[key1])

class_sorted_top_k_dict = {}

for key, value in class_top_k_dict.items():
    sorted_fg = sorted(value["fg"].items(), key=lambda item: item[1], reverse=True)
    class_sorted_top_k_dict[key] = {"name": value["name"], "fg": dict(sorted_fg)}#, "bg": dict(sorted_bg)}

top_k = 10

idx = 1

for key, val in class_sorted_top_k_dict.items():
    idx = idx + 1
    count = 0
    for key1, value1 in val["fg"].items():
        if count==top_k:
            break
        count = count + 1

final_class_dict = {}
for key, value in class_sorted_top_k_dict.items():
    ori_list = []
    temp_class_dict = {}
    for key1 in value["fg"]:
        ori_list.append(key1)

    collapsed_list = get_collapsed_categories(ori_list)

    final_img_dict = {"idx": [], "count":[], "score":[], "col_list":[], "image": []}

    for idx, cl in enumerate(collapsed_list):
        temp_count = 0
        temp_score = 0.0
        temp_img_list = []
        for obj in cl:
            temp_count = temp_count + value["fg"][obj]
            temp_score = temp_score + (value["fg"][obj]*1.0)/(pivot_table.loc[key, obj]*1.0)
            temp_img_list = temp_img_list + obj_image_dict[key][obj]

        final_img_dict["idx"].append(idx)
        final_img_dict["count"].append(temp_count)
        final_img_dict["score"].append(temp_score)
        final_img_dict["image"].append(temp_img_list)
        final_img_dict["col_list"].append(cl)
    final_class_dict[key] = final_img_dict

sorted_final_class_dict = {}

for key in final_class_dict:
    sorted_temp_class_dict= sorted(zip(final_class_dict[key]['score'], final_class_dict[key]['count'], final_class_dict[key]['idx'], final_class_dict[key]['image'], final_class_dict[key]['col_list']), reverse=True)

    # Separate the sorted values back into their respective lists
    sorted_scores, sorted_counts, sorted_indices, sorted_images, sorted_col_list = zip(*sorted_temp_class_dict)

    # Create the sorted dictionary
    sorted_temp_class_dict = {
        'idx': list(sorted_indices),
        'score': list(sorted_scores),
        'count': list(sorted_counts),
        'col_list': list(sorted_col_list),
        'image': list(sorted_images)
    }

    sorted_final_class_dict[key] = sorted_temp_class_dict

def generate_random_combination(images, k):
    return random.sample(images, k)

tc = 0
max_lim = 35

for key, value in tqdm(sorted_final_class_dict.items(),desc="Final image save", total=len(sorted_final_class_dict)):
    count = 0

    class_path = base_img_path + labels_dict[key] + "/"
    os.makedirs(class_path, exist_ok=True)

    for idx, cnt in enumerate(value["count"]):
        if count >=max_lim:
            break
        
        temp = 0
        if idx == 0:
            temp = min(cnt, 15)
        else:
            temp = min(cnt, 10)

        req = max_lim - count

        if req<=0:
            break

        if temp > req:
            temp = req

        assert temp <= len(value["image"][idx]), "Not enough images"

        for img in rand_list:
            image = Image.open(os.path.join(directory_path,img))
            image_path = os.path.join(class_path, img)
            image.save(image_path, format='JPEG')
            image.close()
        
        count = count + temp