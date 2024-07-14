# ASPIRE
Code for the paper: [ASPIRE: Language-Guided Data Augmentation for Improving Robustness
Against Spurious Correlations](https://arxiv.org/pdf/2308.10103)

![Proposed Methodology](./assets/aspire.png)

## Image Captioning

```
python image_captioning/caption_images.py \
-i <input_image_file_path> \
-o <output_file_path>
```

Each line in the input file should contain path to an image we want to caption.

## Extracting objects and backgrounds from captions.
We use the following prompt with GPT-4 to identify forground objects, backgrounds and suggestions for alternate backgrounds for captions generated for images in the above step:

```
I will provide you with a list of tuples. Each tuple in the list has 2 items: the first is a caption of an image and the second is the label of the image. For each, you will have to return a JSON with 3 lists. One list should be the list of all phrases from the caption that are objects that appear in the foreground of the image but ignore objects that correspond to the actual label (the label for the phrase might not be present exactly in the caption) (named ’foreground’). The second list should have the single predominant background of the image to the foreground objects (named ’background’). If you do not find a phrase that corresponds to the background, return an empty list for the background. The third is an alternative background for the image, an alternative to the background you suggested earlier (named ’alt’). Here are some examples which also show the format in which you need to return the output. Please just return the JSON in the following format: {"foreground": ["woman", "plaid kilt"], "background": ["forest"], "alt": ["city streets"]}, {"foreground": ["dog", "coke can"], "background": ["bed"], "alt": ["playground"]}  and here is the caption: {caption}. 
```



## 🌻 Acknowledgement  
We use the code from the following repositories: [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything), [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything) and [Instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix).

Please cite the above repositories if you find their code useful.

## 🔏 Citation    
```
@misc{ghosh2024aspirelanguageguideddataaugmentation,
      title={ASPIRE: Language-Guided Data Augmentation for Improving Robustness Against Spurious Correlations}, 
      author={Sreyan Ghosh and Chandra Kiran Reddy Evuru and Sonal Kumar and Utkarsh Tyagi and Sakshi Singh and Sanjoy Chowdhury and Dinesh Manocha},
      year={2024},
      eprint={2308.10103},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2308.10103}, 
}
```