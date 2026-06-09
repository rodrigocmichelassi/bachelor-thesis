## Bridging Text and Retinal Imaging: A Multimodal Search Approach

This project describes how to use CLIP and SIGLIP for retinal image-text alignment.

In this document, we intend to describe how to properly execute the scripts, what each of them do and use them for model training and image-searching from natural language.

### Folders structure 

**Fill this later**

### Data Generation

`./src/notebooks/data-generation.ipynb`: This notebook is responsible for the analysis on the BRSET dataset, its data distribution regarding several signals we have for the dataset and defining how to generate the captions for each image.

`./src/scripts/data-generation.py`: This Python script is used to generate captions for the BRSET dataset. It has two main execution modes
- Prepare Dataframe: gather signals from the BRSET dataset and the YOLO models for Optic Disc and Macula detection, and generate a `.csv` file with annotations for the data.
- Generate Captions: use the `.csv` file prepared in the previous mode, together with a local LLM, to generate captions for each image. In this dataset, we leverage three possible types of captions:
  - Anatomy captions: focus on the disposal of the Optic Disc and Macula within the image.
  - Quality captions: whether the image has a focus/illumination issue.
  - Disease captions: whether the retina is healthy or present a specific disease.
     
To prepare the dataset, run:

```sh
cd scripts
python data-generation.py \
  --prepare_df {0,1} \
  --plot_distributions {0,1} \
  --generate_captions {0,1}
```
