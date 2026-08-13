# Returns natural language optic disc side in the image description
# Parameters
# - od_side: [0,1] value describing the side of the optic disc in the image
def get_od_side(od_side):
    if od_side < 0.4:
        return 'the optic disc is in the left side of the retinal image'
    elif od_side < 0.6:
        return 'the optic disc is centered in the retinal image'
    return 'the optic disc is in the right side of the retinal image'

# Returns natural language eye side from which the photo was taken
# Parameters
# - exam_eye: from which eye was the exam taken, according to BRSET annotation
def get_exam_eye(exam_eye):
    if exam_eye == 1:
        return 'right eye'
    return 'left eye'

# Returns natural language macula distance to temporal edge description
# Parameters
# - temporal_distance: the distance from the macula to the temporal edge, in pixels
def get_temporal_distance(td):
    if td < 900:
        return "macula located very close to the temporal retinal boundary"
    elif td < 1050:
        return "macula positioned near the temporal retinal boundary"
    elif td < 1200:
        return "macula occupying a typical temporal position"
    elif td < 1350:
        return "macula slightly displaced from the temporal retinal boundary"
    return "macula occupying a relatively central position within the retinal field"

# Returns natural language optic disc distance to nasal edge description
# Parameters
# - nasal_distance: the distance from the optic disc to the nasal edge, in pixels
def get_nasal_distance(nd):
    if nd < 200:
        return "optic disc located very close to the nasal retinal boundary"
    elif nd < 300:
        return "optic disc positioned near the nasal retinal boundary"
    elif nd < 450:
        return "optic disc occupying a typical nasal position"
    elif nd < 600:
        return "optic disc slightly displaced from the nasal retinal boundary"
    return "optic disc located farther from the nasal retinal boundary"

# Returns natural language optic disc size description
# Parameters
# - dd: the optic disc size in the image, in pixels
def get_disc_diameter(dd):
    if dd < 300:
        return 'small optic disc diameter'
    elif dd < 380:
        return 'normal-sized optic disc diameter'
    return 'large optic disc diameter'

# Returns natural language optic disc-macula angle description
# Parameters
# - angle: the angle between the optic disc and the macula from the detections
def get_od_macula_angle(angle):
    if angle < 7:
        return "optic disc and macula appear nearly horizontally aligned"
    elif angle < 13:
        return "optic disc and macula show a moderate vertical offset"
    return "the anatomical axis between the optic disc and macula is larger than usual"

# Generates a prompt for retinal image anatomy caption generation
# Parameters
# - row: processed brset dataframe row with anatomy annotations
def get_anatomy_information(row):
    angle = get_od_macula_angle(row['od_fovea_angle_degrees'])
    disc_diameter = get_disc_diameter(row['od_diameter'])
    nasal_distance = get_nasal_distance(row['nasal_distance'])
    temporal_distance = get_temporal_distance(row['temporal_distance'])
    exam_eye = get_exam_eye(row['exam_eye'])
    od_side = get_od_side(row['od_side'])

    return angle, disc_diameter, nasal_distance, temporal_distance, exam_eye, od_side

# Returns natural language macula visibiltiy description on the image
# based on the YOLO detection confidence
# Parameters
# - fovea_confidence: detection confidence for macula/fovea
def get_macula_visibility(fovea_confidence):
    if fovea_confidence < 0.4:
        return 'low visibility'
    elif fovea_confidence < 0.7:
        return 'partially visible'
    elif fovea_confidence < 0.9:
        return 'good visibility'
    return 'clearly visible'

# Returns natural language focus condition on the image
# Parameters
# - focus_value: 1 for good focus, 2 for bad focus
def get_focus_information(focus_value):
    if focus_value == 2:
        return 'image with focus problem'
    return 'image with adequate focus'

# Returns natural language illumination condition on the image
# Parameters
# - illumination_value: 1 for good illumination, 2 for bad illumination
def get_illumination_information(illumination_value):
    if illumination_value == 2:
        return 'image with bad illumination'    # extend this to have better analysis on illumination
    return 'image with adequate illumination'

# Generates a prompt for retinal image quality caption generation
# Parameters
# - row: processed brset dataframe row with quality annotations
def generate_quality_prompt(row):
    macula_visibility = get_macula_visibility(row['fovea_detection_confidence'])
    focus_information = get_focus_information(row['focus'])
    illumination_information = get_illumination_information(row['iluminaton'])

    prompt = f"""You are an ophthalmology assistant.
        Generate a short sentence describing image quality.
        Focus on:
        - focus
        - illumination
        - visibility of retinal structures
        Do not mention diseases.
        Do not mention anatomy.
        Return only the sentence, no extra texts.
        Try to provide a few variability on how the sentence is produced and how
        you mention macular visibility. You may use similar words for this, such as
        fovea.
        
        Image findings:
        - macula visibility: {macula_visibility}
        - image focus: {focus_information}
        - image illumination: {illumination_information}.

        Return exactly one sentence.
        """

    return prompt

# Generates a prompt for retinal image disease caption generation
# Parameters
# - row: processed brset dataframe row with diseases annotations
def generate_disease_prompt(row):
    disease_caption_columns = ['healthy', 'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd', 'vascular_occlusion', 'hypertensive_retinopathy', 'drusens', 'hemorrhage', 'myopic_fundus', 'increased_cup_disc']

    # list with the name of the diseases present in the image
    findings = [
        col
        for col in disease_caption_columns
        if row[col] == 1
    ]

    if len(findings) == 0:
        findings = ["healthy"]

    metadata = "\n".join(f"- {finding}" for finding in findings)
    print(metadata)
    
    prompt = f"""
        Generate a short sentence describing the retinal findings for a given image, to describe the present
        retinal diseases or if the retina is healthy. 
        Only use the diseases given in the metadata; if the retina is healthy, metadata will contain 'healthy'.
        Do not infer diagnoses from findings.
        Describe only the findings explicitly provided.

        - Do not interpret findings.
        - Do not infer diagnoses or diseases not explicitly provided.
        - Do not mention diseases not explicitly provided.
        - Describe only the findings listed below.
        - Mention only information explicitly provided.
        - Do not explain findings.
        - Do not interpret findings.
        - Do not suggest, indicate, imply, correlate with, or raise suspicion for any disease.
        - Do not use synonyms for diseases.
        - Do not rephrase disease names.
        - Do not describe severity for diseases unless explicitly provided.

        Image findings:
        {metadata}

        Return exactly one sentence.
        """

    return prompt

# Generates a prompt for retinal image anatomy caption generation
# Parameters
# - row: processed brset dataframe row with anatomy annotations
def generate_anatomy_prompt(row):
    ag, _, nd, td, ee, _ = get_anatomy_information(row)

    prompt = f"""
        Generate a single anatomical description of a retinal image.

        Anatomical observations:
        - {ag}
        - {nd}
        - {td}
        - {ee}

        Rules:
        - Do not copy the observations given.
        - Combine and paraphrase the observations into a single sentence.
        - Vary the wording and sentence structure across images.
        - Mention only anatomical observations.
        - Do not mention diseases or suggests any diagnosis.
        - Do not mention measurements, only descriptions.
        - Use 2 or 3 observations; the sentence must be short.
        - Use only the observations provided.
        - Do not infer additional anatomical relationships.
        - Do not explain findings.
        - Do not describe causes or implications.
        - Do not introduce information not explicitly present in the observations.
        - The caption should be a direct visual description of the retinal photograph.
        - Use at most 15 words.
        
        Return exactly one sentence.
        """

    return prompt

# Generates a prompt for retinal image caption generation
# containing anatomy, quality and disease information
# Parameters
# - row: processed brset dataframe row with retinal annotations
def generate_combined_prompt(row):
    ag, _, nd, td, ee, _ = get_anatomy_information(row)

    macula_visibility = get_macula_visibility(
        row['fovea_detection_confidence']
    )

    focus_information = get_focus_information(
        row['focus']
    )

    illumination_information = get_illumination_information(
        row['iluminaton']
    )

    disease_caption_columns = [
        'healthy',
        'diabetic_retinopathy',
        'macular_edema',
        'scar',
        'nevus',
        'amd',
        'vascular_occlusion',
        'hypertensive_retinopathy',
        'drusens',
        'hemorrhage',
        'myopic_fundus',
        'increased_cup_disc'
    ]

    findings = [
        col
        for col in disease_caption_columns
        if row[col] == 1
    ]

    if len(findings) == 0:
        findings = ["healthy"]

    disease_metadata = "\n".join(
        f"- {finding}" for finding in findings
    )

    prompt = f"""
    Generate a single caption describing a retinal photograph.

    Disease findings:
    {disease_metadata}

    Anatomical observations:
    - {ag}
    - {nd}
    - {td}
    - {ee}

    Image quality observations:
    - {macula_visibility}
    - {focus_information}
    - {illumination_information}

    Rules:
    - Combine disease, anatomy, and image quality information.
    - Mention only information explicitly provided.
    - Do not infer diagnoses or diseases not explicitly provided.
    - Describe findings only.
    - Do not explain findings.
    - Do not interpret findings.
    - Do not suggest, indicate, imply, correlate with, or raise suspicion for any disease.
    - Do not use synonyms for diseases.
    - Do not rephrase disease names.
    - Do not describe severity for diseases unless explicitly provided.
    - Do not mention numerical measurements.
    - Keep the caption concise.
    - Write naturally as if describing a retinal fundus photograph.
    - Use no more than 25 words.
    - Prioritize disease findings when present.
    - Only include quality observations if there is an explicit quality issue in the image.
        - Omit macula visibility, focus and illumination information whenever it is 'fine', 'good', 'clear', 'adequate'.

    Return exactly one sentence.
    """

    return prompt

# Takes a prompt to generate captions for an image
# Parameters
# - prompt: the given prompt to generate the caption (anatomy, disease, quality)
# - system_prompt: general prompt given for all captions
# - model: natural language model instance
# - tokenizer: which tokenizer to use
def generate_image_caption(prompt, system_prompt, model, tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
    return response

if __name__ == '__main__':
    pass