import streamlit as st
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import numpy as np
import supervision as sv
import os
import albumentations as A
import cv2
from transformers import AutoConfig
from Florence2.preprocessor_florence2 import Florence2Processor
import yaml

# Load Model and Processor
@st.cache_resource
def load_model():
    CHECKPOINT = "~/Checkpoints/AD-KD-MICCAI25/checkpoints/models"
    REVISION = 'refs/pr/6'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_model = AutoConfig.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
    config_model.vision_config.model_type = "davit"
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True, config=config_model, revision=REVISION).to(DEVICE)

    processor = Florence2Processor.from_pretrained("./Florence2")
    processor.image_processor.size = 512
    processor.image_processor.crop_size = 512
    return model, processor, DEVICE

model, processor, DEVICE = load_model()

# Load definition files
@st.cache_resource
def load_definitions():
    vindr_path = '../configs/vindr_definition.yaml'
    padchest_path = '../configs/padchest_definition.yaml'
    with open(vindr_path, 'r') as file:
        vindr_definitions = yaml.safe_load(file)
    with open(padchest_path, 'r') as file:
        padchest_definitions = yaml.safe_load(file)
    return vindr_definitions, padchest_definitions

vindr_definitions, padchest_definitions = load_definitions()

dataset_options = {"Vindr": vindr_definitions, "PadChest": padchest_definitions}

# Define the transform
def apply_transform(image, size_mode=512):
    pad_resize_transform = A.Compose([
        A.LongestMaxSize(max_size=size_mode, interpolation=cv2.INTER_AREA),
        A.PadIfNeeded(
            min_height=size_mode, min_width=size_mode, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        ),
        A.Resize(height=512, width=512, interpolation=cv2.INTER_AREA),
    ])
    image_np = np.array(image)
    transformed = pad_resize_transform(image=image_np)
    return transformed["image"]

# Streamlit UI
st.title("Inference GUI")

# Image input: file upload or file path
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
with col2:
    image_path = st.text_input(
        "Or enter the image file path",
        value="/vol/ciamspace/datasets/X-ray/vindr-cxr/processed/images_512/f1eb2216d773ced6330b1f31e18f04f8.png"
    )

# Load image from file or path
image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif image_path:
    try:
        if os.path.isfile(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            st.error("File not found. Please check the file path.")
    except FileNotFoundError:
        st.error("File not found. Please check the file path.")

# Option to apply transform
apply_transform_option = st.checkbox("Apply Transform (Resize and Pad to 512x512)", value=False)
if image is not None:
    if apply_transform_option:
        transformed_image = apply_transform(image, size_mode=512)
        image = Image.fromarray(transformed_image)
        st.image(image, caption="Transformed Image (512x512)", width=600)
    else:
        st.image(image, caption=f"Input Image: {image_path}", width=600)

# Dataset selection
dataset_choice = st.selectbox("Select Dataset", options=list(dataset_options.keys()))
disease_options = list(dataset_options[dataset_choice].keys())
disease_choice = st.selectbox("Select Disease", options=disease_options)

definition = dataset_options[dataset_choice][disease_choice]

# Result button and inference
if st.button("Run Inference"):
    if image is None:
        st.error("Please upload an image or enter a valid file path.")
    else:
        det_obj = f"{disease_choice} means {definition}."
        st.write(f"**Definition:** {definition}")
        
        prompt = f"Locate the phrases in the caption: {det_obj}."
        prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{prompt}"

        # Preprocess image
        np_image = np.array(image)
        inputs = processor(
            text=[prompt],
            images=[np_image],
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        # Model inference
        with st.spinner("Processing..."):
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
            generated_text = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]

            # Post-process results
            predictions = processor.post_process_generation(
                generated_text, 
                task="<CAPTION_TO_PHRASE_GROUNDING>", 
                image_size=np_image.shape[:2]
            )

            # Convert predictions to detections for visualization
            detection = sv.Detections.from_lmm(
                sv.LMM.FLORENCE_2, 
                predictions, 
                resolution_wh=np_image.shape[:2]
            )

            # Annotate the image with predictions
            bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
            label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
            image_with_predictions = bounding_box_annotator.annotate(np_image.copy(), detection)
            image_with_predictions = label_annotator.annotate(image_with_predictions, detection)
            annotated_image = Image.fromarray(image_with_predictions.astype(np.uint8))

            # Display the results with adjusted size
            st.image(annotated_image, caption="Inference Results", width=600)
            st.write("**Generated Text:**", generated_text)