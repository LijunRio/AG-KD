import requests
from PIL import Image
from numpy import asarray
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import re
from multi_task_dataset import VindrDataset  

def inference_radvlm(model, processor, image, prompt, chat_history=None, max_new_tokens=1500):
    """
    Generate a response using RadVLM in either single-turn or multi-turn mode.

    Args:
        model: The RadVLM model.
        processor: The processor for RadVLM (provides apply_chat_template and tokenization).
        image: A PIL Image or NumPy array representing the input image.
        prompt: The user prompt for this turn.
        chat_history: A list of (user_msg, assistant_msg) tuples representing the conversation so far.
                      If None or empty, single-turn mode is used. Even in single-turn mode, 
                      this function returns chat_history so that you can continue in subsequent turns.
        max_new_tokens: The maximum number of new tokens to generate.

    Returns:
        response (str): The assistant's response for this turn.
        chat_history (list): The updated chat_history including this turn's (prompt, response).
    """

    # Initialize chat history if not provided
    if chat_history is None:
        chat_history = []

    # Build the chat history 
    conversation = []
    for idx, (user_text, assistant_text) in enumerate(chat_history):
        if idx == 0:
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image"},
                ],
            })
        else:
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                ],
            })
        conversation.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text},
            ],
        })

    # Add the current user prompt
    if len(chat_history) == 0:
        # First turn includes the image
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        })
    else:
        # Subsequent turns without the image
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        })

    # Apply the chat template to create the full prompt
    full_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Prepare model inputs
    inputs = processor(images=image, text=full_prompt, return_tensors="pt", padding=True).to(
        model.device, torch.float16
    )

    # Generate the response
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode the output
    full_response = processor.decode(output[0], skip_special_tokens=True)
    response = re.split(r"(user|assistant)", full_response)[-1].strip()

    # Update chat history
    chat_history.append((prompt, response))

    return response, chat_history



# import torch
# from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
# from PIL import Image
# import requests
# from io import BytesIO
# import numpy as np

# #  Initialize the model and processor
# model_id = "KrauthammerLab/RadVLM"
# model = LlavaOnevisionForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     low_cpu_mem_usage=True, 
# ).to('cuda')  # Use 'cuda' if GPU is available, else 'cpu'

# processor = AutoProcessor.from_pretrained(model_id)

# image_url = "https://prod-images-static.radiopaedia.org/images/29923576/fed73420497c8622734f21ce20fc91_gallery.jpeg"
# image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

# # Initialize chat history
# chat_history = []

# # First user prompt with image from URL
# user_prompt_1 = "Where is the pneumothorax located on the image?"
# response_1, chat_history = inference_radvlm(model, processor, image, user_prompt_1, chat_history)

# print("RadVLM:", response_1)

# from PIL import Image, ImageDraw
# # Extract bounding box coordinates from the response
# # Assuming the output format is: "The pneumothorax is situated at [0.89, 0.68, 0.95, 0.78]"
# bbox_str = re.search(r"\[([0-9\.]+), ([0-9\.]+), ([0-9\.]+), ([0-9\.]+)\]", response_1)
# if bbox_str:
#     x1, y1, x2, y2 = map(float, bbox_str.groups())

#     # Image dimensions
#     width, height = image.size

#     # Convert the normalized coordinates to pixel values
#     x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

#     # Draw the bounding box on the image
#     draw = ImageDraw.Draw(image)
#     draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

#     # Save the image with the bounding box to a file
#     output_file_path = "pneumothorax_bounding_box.jpg"
#     image.save(output_file_path)
#     print(f"Image with bounding box saved to {output_file_path}")

# else:
#     print("No bounding box found in the response.")

# # # Second user prompt, continuing the conversation
# # user_prompt_2 = "Is there something concerning in the lungs area?"
# # response_2, chat_history = inference_radvlm(model, processor, image, user_prompt_2, chat_history)

# # print("RadVLM:", response_2)

# # # Third user prompt
# # user_prompt_3 = "What about the cardiac silhouette? Is it normal?"
# # response_3, chat_history = inference_radvlm(model, processor, image, user_prompt_3, chat_history)

# # print("Assistant:", response_3)


import torch
import numpy as np
import pandas as pd
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from torch.utils.data import Dataset
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import ast
import re
import os

class VindrDataset(Dataset):
    def __init__(self, img_root, annotation_csv, split='test', data_pct=1.0, transform=None):
        self.img_root = img_root
        self.transform = transform

        if split not in ['train', 'test', 'validate']:
            raise ValueError(f"Invalid split: {split}. Expected one of ['train', 'test', 'validate'].")
        
        if not (0 < data_pct <= 1):
            raise ValueError(f"data_pct should be in the range (0, 1], got {data_pct}")
        
        self.annotations = pd.read_csv(annotation_csv)
        if split == 'train':
            self.annotations = self.annotations[(self.annotations['split'] == 'train') | (self.annotations['split'] == 'validate')].reset_index(drop=True)
        else:
            self.annotations = self.annotations[self.annotations['split'] == split].reset_index(drop=True)

        if self.annotations.empty:
            raise ValueError(f"No data available for split: {split}")

        if data_pct < 1.0:
            sampled_indices = np.random.choice(len(self.annotations), size=int(len(self.annotations) * data_pct), replace=False)
            self.annotations = self.annotations.iloc[sampled_indices].reset_index(drop=True)
        print(f"Loaded {len(self.annotations)} samples for split: {split} with data_pct: {data_pct}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_id = self.annotations.iloc[idx]['image_id']
        img_path = os.path.join(self.img_root, f"{img_id}.png") 
        image = np.array(Image.open(img_path).convert("RGB"))
        
        class_name = self.annotations.iloc[idx]['class_name'].replace('vindrcxr/', '')
        if class_name == 'Nodule/Mass':
            class_name = 'Nodule or Mass'
        class_name = class_name.lower()
        boxes = ast.literal_eval(self.annotations.iloc[idx]['bboxes'])

        return {
            'image': image,
            'boxes': boxes,
            'label': class_name
        }
    
import torch
import lightning as L  # PyTorch Lightning

model_id = "KrauthammerLab/RadVLM"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to('cuda') 


processor = AutoProcessor.from_pretrained(model_id)

test_dataset = VindrDataset(img_root="/vol/ciamspace/datasets/X-ray/vindr-cxr/processed/images_512/", annotation_csv="../annotations/vindr_dataset.csv", split="test", data_pct=1.0)


CLASSES = ['Pleural thickening', 'Aortic enlargement', 'Pulmonary fibrosis', 'Cardiomegaly', 'Nodule or Mass', 'Lung Opacity', 'Other lesion', 'Pleural effusion', 'ILD', 'Infiltration', 'Calcification', 'Consolidation', 'Atelectasis', 'Rib fracture', 'Mediastinal shift', 'Enlarged PA', 'Pneumothorax', 'Emphysema', 'Lung cavity', 'Lung cyst', 'Clavicle fracture', 'Edema']
CLASSES = [cls.lower() for cls in CLASSES]

def convert_to_format(pred_boxes, pred_labels):
    # {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': [[330.4960021972656, 164.09600830078125, 391.9360046386719, 209.6640167236328]], 'labels': ['pulmonary fibrosis']}}
    formatted_output = {'<CAPTION_TO_PHRASE_GROUNDING>': {'bboxes': pred_boxes, 'labels': pred_labels}}
    return formatted_output

import supervision as sv
import json

targets = []
predictions = []
bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
count = 0
import yaml
definition = yaml.safe_load(open('/u/home/lj0/Code/florence2/preprocess/vindr/definition_shorter.yaml'))
for data in tqdm(test_dataset, desc="Calculating mAP"):
    image = data['image']
    true_boxes = data['boxes']
    class_name = data['label']
    gt_cls_list = [class_name]*len(true_boxes)
    # import pdb; pdb.set_trace()
    gt_answer = convert_to_format(true_boxes, gt_cls_list)
    gt = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, gt_answer, resolution_wh=image.shape)
    gt.class_id = np.array([CLASSES.index(class_name) for class_name in gt['class_name']])
    def_obj = definition[class_name]

    user_prompt = f"Locate and describe the {class_name} in this image."
    response, _ = inference_radvlm(model, processor, image, user_prompt)
    
    bbox_strs = re.findall(r"\[([0-9\.]+), ([0-9\.]+), ([0-9\.]+), ([0-9\.]+)\]", response)
    pred_boxes = []
    if bbox_strs:
        width, height = Image.fromarray(image).size
        for bbox_str in bbox_strs:
            x1, y1, x2, y2 = map(float, bbox_str)
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
            pred_boxes.append([x1, y1, x2, y2])
    # print(pred_boxes)
    # import pdb; pdb.set_trace()
    pred_cls = [class_name]*len(pred_boxes)
    pred_answer = convert_to_format(pred_boxes, pred_cls)
    pred = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, pred_answer, resolution_wh=image.shape)
    pred.class_id = np.array([CLASSES.index(class_name) for class_name in pred['class_name']])
    pred.confidence = np.ones(len(pred))
    # import pdb; pdb.set_trace()
    targets.append(gt)
    predictions.append(pred)
    # if count % 20 == 0:
    #     image_with_ground_truth = bounding_box_annotator.annotate(image.copy(), gt)
    #     image_with_ground_truth = label_annotator.annotate(image_with_ground_truth, gt)
    #     pred['class_name'] = ['pred_' + class_name] * len(pred['class_name'])
    #     image_with_predictions = bounding_box_annotator.annotate(image_with_ground_truth.copy(), pred)
    #     image_with_predictions = label_annotator.annotate(image_with_predictions, pred)
    #     image_with_predictions = Image.fromarray(image_with_predictions.astype(np.uint8))
    #     image_with_predictions.save(f"./radvlm_outputs/predictions_{count}.jpg")
    # count += 1
   
res_csv = []
for i in range (len(predictions)):
    pred = predictions[i]
    tgt = targets[i]
    # import pdb; pdb.set_trace()
    tgt_bbox = tgt.xyxy.tolist()
    pred_bbox = pred.xyxy.tolist()
    tgt_class_id = tgt.class_id.tolist()
    pred_class_id = pred.class_id.tolist()

    tgt_class_name = tgt['class_name'].tolist()
    pred_class_name = pred['class_name'].tolist() if pred['class_name'] is not None else [[]]
    row = {'tgt_bbox':tgt_bbox, 'pred_bbox':pred_bbox, 'tgt_class_id':tgt_class_id, 'pred_class_id':pred_class_id, 'tgt_class_name':tgt_class_name, 'pred_class_name':pred_class_name}
    res_csv.append(row)
import pandas as pd
res_csv = pd.DataFrame(res_csv)
res_csv.to_csv('../res/radVLM_vindr_res.csv', index=False)


mean_average_precision = sv.MeanAveragePrecision.from_detections(
    predictions=predictions, 
    targets=targets
)


print("mAP_50_95:", mean_average_precision.map50_95)
print("mAP_50:", mean_average_precision.map50)
print("mAP_75:", mean_average_precision.map75)

# Save the results to a JSON file
results = {
    "mAP_50_95": mean_average_precision.map50_95,
    "mAP_50": mean_average_precision.map50,
    "mAP_75": mean_average_precision.map75,
}



