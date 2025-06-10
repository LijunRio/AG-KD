import shutil
import os
# Clear the /tmp/hf_cache directory
hf_cache_path = "/tmp/hf_cache"
if os.path.exists(hf_cache_path):
    shutil.rmtree(hf_cache_path)



import os
os.environ["HF_HOME"] = "/tmp/hf_cache" 

import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import yaml
import cv2
import supervision as sv
import albumentations as A

import matplotlib.pyplot as plt



def load_model():
    model_name = "RioJune/AG-KD"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


    return model, processor, device

def apply_transform(image, size=512):
    transform = A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        A.Resize(height=size, width=size)
    ])
    return transform(image=np.array(image))["image"]

def load_definitions():
    with open("configs/vindr_definition.yaml") as f1, open("configs/padchest_definition.yaml") as f2:
        vindr = yaml.safe_load(f1)
        padchest = yaml.safe_load(f2)
    return {**vindr, **padchest}

def run(image_path, target, model, processor, device, definitions):
    definition = definitions.get(target)
    if definition is None:
        print(f"[ERROR] Definition not found for target: {target}")
        return

    prompt = f"<CAPTION_TO_PHRASE_GROUNDING>Locate the phrases in the caption: {target} means {definition}."

    image = Image.open(image_path).convert("RGB")
    np_image = apply_transform(image)
    inputs = processor(text=[prompt], images=[np_image], return_tensors="pt", padding=True).to(device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        output_scores=True,
        return_dict_in_generate=True
    )

    transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False)
    generated_text = processor.batch_decode(outputs.sequences, skip_special_tokens=False)[0]

    input_len = inputs.input_ids.shape[1]
    output_len = np.sum(transition_scores.cpu().numpy() < 0, axis=1)
    length_penalty = model.generation_config.length_penalty
    score = transition_scores.cpu().sum(axis=1) / (output_len**length_penalty)
    prob = np.exp(score.cpu().numpy())

    print(f"\n[IMAGE] {image_path}")
    print(f"[TARGET] {target}")
    print(f"[PROBABILITY] {prob[0] * 100:.2f}%")
    print(f"[GENERATED TEXT]\n{generated_text}")

    predictions = processor.post_process_generation(generated_text, task="<CAPTION_TO_PHRASE_GROUNDING>", image_size=np_image.shape[:2])
    detection = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, predictions, resolution_wh=np_image.shape[:2])
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated = label_annotator.annotate(np_image.copy(), detection)
    annotated = bounding_box_annotator.annotate(annotated, detection)

    # Show instead of save
    plt.figure(figsize=(8, 8))
    plt.imshow(annotated)
    plt.axis("off")
    plt.title(f"{target} - {prob[0] * 100:.2f}%")
    plt.show()

if __name__ == "__main__":
    examples = {
        "./examples/26746130963764173994750391023442607773-2_mukhp1.png": "electrical device",
        "./examples/f1eb2216d773ced6330b1f31e18f04f8.png": "pulmonary fibrosis",
        "./examples/fb4dfacc089f4b5550f03f52e706b6f2.png": "cardiomegaly"
    }

    model, processor, device = load_model()
    definitions = load_definitions()

    for image_path, target in examples.items():
        run(image_path, target, model, processor, device, definitions)
