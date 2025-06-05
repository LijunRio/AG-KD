
import torch
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import albumentations as A
from transformers import AutoModelForCausalLM, AutoProcessor


def apply_transform(image, size=512):
    transform = A.Compose([
        A.LongestMaxSize(max_size=size),
        A.PadIfNeeded(min_height=size, min_width=size, border_mode=0, value=(0,0,0)),
        A.Resize(height=size, width=size)
    ])
    return transform(image=np.array(image))["image"]

def run_simple(image_url, target, definition, model, processor, device):
    prompt = f"<CAPTION_TO_PHRASE_GROUNDING>Locate the phrases in the caption: {target} means {definition}."
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
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

    output_len = np.sum(transition_scores.cpu().numpy() < 0, axis=1)
    length_penalty = model.generation_config.length_penalty
    score = transition_scores.cpu().sum(axis=1) / (output_len**length_penalty)
    prob = np.exp(score.cpu().numpy())

    print(f"\n[IMAGE URL] {image_url}")
    print(f"[TARGET] {target}")
    print(f"[PROBABILITY] {prob[0] * 100:.2f}%")
    print(f"[GENERATED TEXT]\n{generated_text}")

if __name__ == "__main__":
    image_url = "https://huggingface.co/spaces/RioJune/AG-KD/resolve/main/examples/f1eb2216d773ced6330b1f31e18f04f8.png"
    target = "pulmonary fibrosis"
    definition = "Scarring of the lung tissue creating a dense fibrous appearance."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "RioJune/AG-KD-merged"

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    run_simple(image_url, target, definition, model, processor, device)
