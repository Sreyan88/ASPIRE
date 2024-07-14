from datasets import load_dataset

dataset = load_dataset("imagenet-1k", split="validation")

labels_of_interest = [537, 379, 785, 795, 837, 433, 416, 602, 706, 746, 655, 810, 890, 981, 801]

labels_dict = {
    537: "dog_sled",
    379: "howler_monkey",
    785: "seat_belt",
    795: "ski",
    837: "sunglasses",
    433: "swimming cap",
    416: "balance beam",
    602: "horizontal bar",
    706: "patio",
    746: "puck",
    655: "miniskirt",
    810: "space bar",
    890: "volleyball",
    981: "baseball player",
    801: "snorkel"
}

filtered_data = dataset.filter(lambda example: example['label'] in labels_of_interest)

print(filtered_data)

filtered_data.save_to_disk("/home/sreyang/scratch.ramanid-prj/acm/filt_imagenet_validation")