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

parser = argparse.ArgumentParser(description="Process an input file and generate an output file.")
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

spur_dataset = pd.read_json(args.wrong_preds_jsonl_path, lines=True)

directory_path = args.spurious_image_saved_path

base_img_path = args.non_spurious_images_save_path

# Get a list of all files in the given directory
file_names = os.listdir(directory_path)

# Filter only the image files (you can adjust the list of allowed extensions as needed)
allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif']
image_files = [file for file in file_names if os.path.splitext(file)[1].lower() in allowed_extensions]

labels_dict = {
    0: "not_blonde",
    1 : "blonde"
}

class_top_k_dict = {}

obj_image_dict = {}

print(f"No of image files: {len(image_files)}")

# Read and process each image
for idx, image_file in tqdm(enumerate(image_files)):
    name = image_file.split(".")[0]
    name_split = name
    image_id = int(name_split)
    idx = spur_dataset.index[spur_dataset['image_id'] == image_id].tolist()[0]
    type = "fg"
    obj = "male" if spur_dataset["Male"][idx] == 1 else "female"
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
    # sorted_bg = sorted(value["bg"].items(), key=lambda item: item[1], reverse=True)

    class_sorted_top_k_dict[key] = {"name": value["name"], "fg": dict(sorted_fg)}#, "bg": dict(sorted_bg)}

top_k = 10

idx = 1

print(len(class_sorted_top_k_dict))

for key, val in class_sorted_top_k_dict.items():
    print(f"For class {idx} : {val['name']}")
    idx = idx + 1
    print("-----------")
    print(f"Foreground top-{top_k} values")
    count = 0
    for key1, value1 in val["fg"].items():
        if count==top_k:
            break
        print(f"{key1}   {value1}")
        count = count + 1

final_class_dict = {}
for key, value in class_sorted_top_k_dict.items():
    ori_list = []
    temp_class_dict = {}
    for key1 in value["fg"]:
        ori_list.append(key1)

    collapsed_list = get_collapsed_categories(ori_list)
    final_img_dict = {"idx": [], "count":[], "image": []}

    for idx, cl in enumerate(collapsed_list):
        temp_count = 0
        temp_img_list = []
        for obj in cl:
            temp_count = temp_count + value["fg"][obj]
            temp_img_list = temp_img_list + obj_image_dict[key][obj]

        final_img_dict["idx"].append(idx)
        final_img_dict["count"].append(temp_count)
        final_img_dict["image"].append(temp_img_list)

    final_class_dict[key] = final_img_dict

sorted_final_class_dict = {}

for key in final_class_dict:
    sorted_temp_class_dict= sorted(zip(final_class_dict[key]['count'], final_class_dict[key]['idx'], final_class_dict[key]['image']), reverse=True)

    # Separate the sorted values back into their respective lists
    sorted_counts, sorted_indices, sorted_images = zip(*sorted_temp_class_dict)

    # Create the sorted dictionary
    sorted_temp_class_dict = {
        'idx': list(sorted_indices),
        'count': list(sorted_counts),
        'image': list(sorted_images)
    }

    sorted_final_class_dict[key] = sorted_temp_class_dict

# print(sorted_final_class_dict)

def generate_random_combination(images, k):
    # print(images)
    return random.sample(images, k)

for key in sorted_final_class_dict:
    print(labels_dict[key])

tc = 0
for key, value in tqdm(sorted_final_class_dict.items(),desc="Final image save", total=len(sorted_final_class_dict)):
    count = 0

    class_path = base_img_path + labels_dict[key] + "/"
    os.makedirs(class_path, exist_ok=True)

    for idx, cnt in enumerate(value["count"]):
        temp = 0
        if key == 0:
            temp = 25
        else:
            temp = 50

        rand_list  = generate_random_combination(value["image"][idx], temp)
        print(f"Rand list len: {len(rand_list)}")

        for img in rand_list:
            image = Image.open(os.path.join(directory_path,img))
            image_path = os.path.join(class_path, img)
            image.save(image_path, format='JPEG')
            image.close()
        
        count = count + temp