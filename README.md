## Bridging Text and Retinal Imaging: A Multimodal Search Approach

This project describes how to use CLIP and SIGLIP for retinal image-text alignment.

In this document, we intend to describe how to properly execute the scripts, what each of them do and use them for model training and image-searching from natural language.

### Folders structure 

**Fill this later**

### Data Generation

`./src/notebooks/data-generation.ipynb`. This notebook is responsible for the analysis and generation of the dataset we will be using.

The core idea is to use BRSET to have initial labels and annotations towards retinal images. From that, we plan to expand the labels available by using a YOLO model that detects the retinal optic disc in images, from pre-defined labels.

This file joins all the important annotations and pass it through a LLM, that will generate natural-language descriptions for each image in the dataset.