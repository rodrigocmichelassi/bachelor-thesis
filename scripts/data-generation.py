import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.yolo_lib import getPredictions, extractInformationFromPred, getEdgeDistanceReal, determineStructureDirection
from src.utils.llm_lib import generate_anatomy_prompt, generate_disease_prompt, generate_quality_prompt, generate_image_caption
from src.config import DISTRIBUTIONS_DIR, OD_MODEL_PATH, FOVEA_MODEL_PATH, BRSET_LABELS_CSV, CAPTIONS_CSV, CLASSIFICATION_CAPTIONS_CSV, RAW_BRSET_LABELS_CSV, RAW_BRSET_IMAGES_DIR 
from ultralytics import YOLO

def get_final_brset_df(filtered_brset_df):
    columns_to_keep = ['image_id', 'exam_eye', 'DR_ICDR', 'focus', 'iluminaton', 
                   'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 
                   'amd', 'vascular_occlusion', 'hypertensive_retinopathy', 'drusens',
                   'hemorrhage', 'myopic_fundus', 'increased_cup_disc', 'quality',
                   'healthy', 'other_abnormalities', 'od_detection_confidence',
                   'od_center_x', 'od_center_y', 'od_diameter', 'od_side',
                   'nasal_distance', 'fovea_detection_confidence', 'fovea_center_x',
                   'fovea_center_y', 'temporal_distance', 'fovea_side',
                   'od_fovea_angle_degrees']

    final_brset_df = filtered_brset_df[columns_to_keep]

    return final_brset_df

def generate_distributions(df):
    save_path = DISTRIBUTIONS_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    plot_distribution(df, 'od_fovea_angle_degrees', 'Angle between the Macula and Optic Disc in Degrees', save_path=save_path / 'angle.png')
    plot_distribution(df, 'od_diameter', 'Optic Disc diameter', save_path=save_path / 'disc_diameter.png')
    plot_distribution(df, 'nasal_distance', 'Distance from Optic Disc to the Nasal Edge', save_path=save_path / 'nasal_dist.png')
    plot_distribution(df, 'temporal_distance', 'Distance from the Macula to the Temporal Edge', save_path=save_path / 'temporal_dist.png')
    plot_distribution(df, 'od_side', 'Optic Disc relative position\n(0: left; 0.5: center; 1: right)', save_path=save_path / 'od_side.png')

def plot_distribution(
    df,
    column_name,
    xlabel,
    bins=40,
    figsize=(8, 5),
    save_path=None
):
    if save_path is None:
        save_path = DISTRIBUTIONS_DIR / 'distribution.png'
        
    data = df[column_name].dropna()

    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, color='tab:blue', density=True, alpha=0.7)
    
    sns.kdeplot(data, color='tab:blue', linestyle='-')

    plt.title(f'Distribution of {xlabel}')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path)

def has_detection(results):
    return len(results[0].boxes) > 0

def mark_failed_detection(df, idx):
    df.loc[idx, 'od_detection_confidence'] = np.nan
    df.loc[idx, 'od_center_x'] = np.nan
    df.loc[idx, 'od_center_y'] = np.nan
    df.loc[idx, 'od_diameter'] = np.nan
    df.loc[idx, 'od_side'] = np.nan
    df.loc[idx, 'nasal_distance'] = np.nan
    df.loc[idx, 'fovea_detection_confidence'] = np.nan
    df.loc[idx, 'fovea_center_x'] = np.nan
    df.loc[idx, 'fovea_center_y'] = np.nan
    df.loc[idx, 'temporal_distance'] = np.nan
    df.loc[idx, 'fovea_side'] = np.nan
    df.loc[idx, 'od_fovea_angle_degrees'] = np.nan

def annotate_detection_results(df, idx, image_path, od_results, fovea_results):
    od_confidence, od_center_coords, disc_diameter = extractInformationFromPred(od_results)
    fovea_confidence, fovea_center_coords, _ = extractInformationFromPred(fovea_results)

    nasal_distance, temporal_distance, od_fovea_angle_degrees = getEdgeDistanceReal(image_path, od_results[0], fovea_results[0])

    od_position = determineStructureDirection(od_results[0], image_path)
    fovea_position = determineStructureDirection(fovea_results[0], image_path)

    # Input obtained data in the dataframe
    df.loc[idx, 'od_detection_confidence'] = od_confidence
    df.loc[idx, 'od_center_x'] = od_center_coords[0]
    df.loc[idx, 'od_center_y'] = od_center_coords[1]
    df.loc[idx, 'od_diameter'] = disc_diameter
    df.loc[idx, 'od_side'] = od_position
    df.loc[idx, 'nasal_distance'] = nasal_distance
    df.loc[idx, 'fovea_detection_confidence'] = fovea_confidence
    df.loc[idx, 'fovea_center_x'] = fovea_center_coords[0]
    df.loc[idx, 'fovea_center_y'] = fovea_center_coords[1]
    df.loc[idx, 'temporal_distance'] = temporal_distance
    df.loc[idx, 'fovea_side'] = fovea_position
    df.loc[idx, 'od_fovea_angle_degrees'] = od_fovea_angle_degrees

def pre_process_brset_df(df):
    diseases_list = ['diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd', 'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 'hemorrhage', 'retinal_detachment', 'myopic_fundus', 'increased_cup_disc', 'other']
    num_diseases_per_row = df[diseases_list].sum(axis=1)

    # Keep only rows with 0 or 1 disease
    filtered_brset_df = df[num_diseases_per_row <= 1]
    filtered_brset_df['healthy'] = (
        filtered_brset_df[diseases_list].sum(axis=1) == 0
    ).astype(int)

    other_abnormalities = ['retinal_detachment', 'other']

    filtered_brset_df['other_abnormalities'] = (
        filtered_brset_df[other_abnormalities].sum(axis=1) > 0
    ).astype(int)

    filtered_brset_df = filtered_brset_df.drop(
        columns=['retinal_detachment', 'other']
    )

    return filtered_brset_df

def get_detection_models(od_model_path, fovea_model_path):
    od_model = YOLO(od_model_path)
    fovea_model = YOLO(fovea_model_path)

    return od_model, fovea_model

def prepare_dataframe(plot_distributions):
    od_model, fovea_model = get_detection_models(str(OD_MODEL_PATH), str(FOVEA_MODEL_PATH))

    brset_df = pd.read_csv(RAW_BRSET_LABELS_CSV)

    filtered_brset_df = pre_process_brset_df(brset_df)

    for idx, row in filtered_brset_df.iterrows():
        # remove this when actually running the algorithm
        print(f'Processing image {idx}')

        img_file = row['image_id'] + '.jpg'
        image_path = os.path.join(RAW_BRSET_IMAGES_DIR, img_file)
        
        # Leverage YOLO Optic Disc and Fovea detection models to gather data
        od_results, fovea_results = getPredictions(imagePath=image_path, odModel=od_model, foveaModel=fovea_model, saveImg=False)

        if not has_detection(od_results) or not has_detection(fovea_results):
            mark_failed_detection(filtered_brset_df, idx)
            continue

        annotate_detection_results(filtered_brset_df, idx, image_path, od_results, fovea_results)

    if plot_distributions:
        generate_distributions(filtered_brset_df)

    final_brset_df = get_final_brset_df(filtered_brset_df)
    final_brset_df.to_csv(BRSET_LABELS_CSV, index=False)

def generate_captions():
    brset_df = pd.read_csv(BRSET_LABELS_CSV)
    brset_df.dropna(inplace=True)

    SYSTEM_PROMPT = """
        You are an ophthalmology assistant.

        Generate three different retinal image captions.

        Each caption should:
        - convey the same information
        - use different wording
        - be medically consistent
        - be concise

        Rules:
        - Use professional medical language.
        - Do not invent findings.
        - Mention only information explicitly provided.
        - Keep captions concise and clinically relevant.
        - Output only the caption.

        Return only one caption.
        """
    
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    captions = []
    for _, row in brset_df.iterrows():
        image_name = row['image_id'] + '.jpg'

        anatomy_prompt = generate_anatomy_prompt(row)
        quality_prompt = generate_quality_prompt(row)
        disease_prompt = generate_disease_prompt(row)
        # combined_prompt = generate_combined_prompt(row)

        disease_caption = generate_image_caption(disease_prompt, SYSTEM_PROMPT, model, tokenizer)
        anatomy_caption = generate_image_caption(anatomy_prompt, SYSTEM_PROMPT, model, tokenizer)
        quality_caption = generate_image_caption(quality_prompt, SYSTEM_PROMPT, model, tokenizer)
        # combined_caption = generate_image_caption(combined_prompt, SYSTEM_PROMPT, model, tokenizer)

        captions.append({
            "image_id": image_name,
            "caption": disease_caption
        })

        captions.append({
            "image_id": image_name,
            "caption": anatomy_caption
        })

        captions.append({
            "image_id": image_name,
            "caption": quality_caption
        })

        # captions.append({
        #     "image_id": image_name,
        #     "caption": combined_caption
        # })

    captions_df = pd.DataFrame(captions)

    captions_df.to_csv(CAPTIONS_CSV, index=False)

def get_classification_caption_from_row(row, n_captions=3):
    DISEASE_TEMPLATES = [
        "This is an image of {disease}.",
        "This retinal image shows {disease}.",
        "A fundus image with signs of {disease}.",
        "Diagnosis: {disease}.",
    ]

    HEALTHY_TEMPLATES = [
        "This is an image of a healthy retina.",
        "This retinal image shows no signs of disease.",
        "A healthy fundus image with no abnormalities.",
        "Diagnosis: healthy retina.",
    ]

    DISEASE_COLUMNS = [
        'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd',
        'vascular_occlusion', 'hypertensive_retinopathy', 'drusens',
        'hemorrhage', 'myopic_fundus', 'increased_cup_disc', 'other_abnormalities'
    ]

    if row.get('healthy') == 1:
        templates = HEALTHY_TEMPLATES
        sampled = random.sample(templates, k=min(n_captions, len(templates)))
        return sampled, 'healthy'

    # find the single active disease column
    active = [d for d in DISEASE_COLUMNS if row.get(d) == 1]

    if len(active) == 0:
        # no disease flagged and not healthy either — shouldn't happen if labels are clean
        raise ValueError(f"No disease or healthy flag found for row: {row.get('image_id')}")
    if len(active) > 1:
        # more than one disease flagged — contradicts the single-label assumption
        raise ValueError(f"Multiple diseases flagged for row: {row.get('image_id')} -> {active}")

    disease_str = active[0].replace('_', ' ')
    templates = random.sample(DISEASE_TEMPLATES, k=min(n_captions, len(DISEASE_TEMPLATES)))

    return [t.format(disease=disease_str) for t in templates], disease_str

def generate_classification_captions():
    brset_df = pd.read_csv(BRSET_LABELS_CSV)
    brset_df.dropna(inplace=True)

    captions = []
    for _, row in brset_df.iterrows():
        image_name = row['image_id'] + '.jpg'
        class_captions, class_name = get_classification_caption_from_row(row)

        captions.append({
            "image_id": image_name,
            "class": class_name,
            "caption_1": class_captions[0],
            "caption_2": class_captions[1],
            "caption_3": class_captions[2]
        })

    captions_df = pd.DataFrame(captions)
    captions_df.to_csv(CLASSIFICATION_CAPTIONS_CSV, index=False)

def main(args):
    if args.prepare_df:
        prepare_dataframe(args.plot_distributions)
    
    if args.generate_captions:
        generate_captions()

    if args.class_captions:
        generate_classification_captions()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SQL operations.")

    parser.add_argument('--prepare_df', type=int, default=1, help='Enrich dataframe with signals')
    parser.add_argument('--plot_distributions', type=int, default=1, help='Whether to plot data distributions or not')
    parser.add_argument('--generate_captions', type=int, default=1, help='Use LLM to generate captions from signals')
    parser.add_argument('--class_captions', type=int, default=0, help='Generate disease captions for classification')
    
    args = parser.parse_args()
    
    main(args)