



import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import os
import numpy as np
import ast
from torch.utils.data import Dataset

# Define the dataset class
class VindrDataset(Dataset):
    def __init__(self, img_root, annotation_csv, split='test', data_pct=1.0, transform=None):
        self.img_root = img_root
        self.transform = transform

        # Check dataset split type
        if split not in ['train', 'test', 'validate']:
            raise ValueError(f"Invalid split: {split}. Expected one of ['train', 'test', 'validate'].")
        
        # Check data percentage
        if not (0 < data_pct <= 1):
            raise ValueError(f"data_pct should be in the range (0, 1], got {data_pct}")
        
        self.annotations = pd.read_csv(annotation_csv)
        if split == 'train':
            self.annotations = self.annotations[(self.annotations['split'] == 'train') | (self.annotations['split'] == 'validate')].reset_index(drop=True)
        else:
            self.annotations = self.annotations[self.annotations['split'] == split].reset_index(drop=True)

        if self.annotations.empty:
            raise ValueError(f"No data available for split: {split}")

        # Sample data based on the percentage
        if data_pct < 1.0:
            sampled_indices = np.random.choice(len(self.annotations), size=int(len(self.annotations) * data_pct), replace=False)
            self.annotations = self.annotations.iloc[sampled_indices].reset_index(drop=True)
        print(f"Loaded {len(self.annotations)} samples for split: {split} with data_pct: {data_pct}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load image
        img_id = self.annotations.iloc[idx]['image_id']
        img_path = os.path.join(self.img_root, f"{img_id}.png")
        # image = np.array(Image.open(img_path).convert("RGB"))
        image = Image.open(img_path).convert("RGB")
        
        # Get true labels and bounding boxes
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
    

import matplotlib.patches as patches
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# Load model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/maira-2", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/maira-2", trust_remote_code=True)
# import pdb; pdb.set_trace()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

# Load Vindr dataset
img_root = "/home/jun/datasets/vindr/images_512/images_512/"
annotation_csv = "../annotations/vindr_dataset.csv"
dataset = VindrDataset(img_root, annotation_csv, split="test", data_pct=1.0)

# Store the ground truth and predicted bounding boxes
gt_bboxes = []
pred_bboxes = []

# Iterate through the dataset and test
for sample in tqdm(dataset):
    image = sample['image']
    label = sample['label']
    boxes = sample['boxes']
    
    # Prepare the input for MAIRA-2
    processed_inputs = processor.format_and_preprocess_phrase_grounding_input(
        frontal_image=image,
        phrase=label,
        return_tensors="pt",
    )

    # Move input to the correct device
    processed_inputs = {key: val.to(device) for key, val in processed_inputs.items()}

    with torch.no_grad():
        # Generate the prediction
        output_decoding = model.generate(
            **processed_inputs,
            max_new_tokens=150,
            use_cache=True,
        )

    # Decode the output and extract the prediction
    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
    # import pdb; pdb.set_trace()
    print('decoded_text: ', decoded_text)
    if not decoded_text.startswith("<obj>"):
        decoded_text = "<obj>" + decoded_text
    if not decoded_text.endswith("</obj>"):
        decoded_text = decoded_text + "</obj>"

    
    prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
    # import pdb; pdb.set_trace()
    # Parse the prediction result
    pred_class_name, pred_bboxes_coords = prediction[0]  # Predicted class and bounding box
    # gt_bboxes.append(boxes)
    # pred_bboxes.append(pred_bboxes_coords)

    # Adjust the predicted bounding boxes for the original image size
    original_size = image.size  # Get the original image size (width, height)
    if not pred_bboxes_coords:
        adjusted_pred_bboxes = []
    else:
        adjusted_pred_bboxes = [
            processor.adjust_box_for_original_image_size(
                bbox, width=original_size[0], height=original_size[1]
            )
            for bbox in pred_bboxes_coords if bbox is not None  # 过滤 None
        ]
        adjusted_pred_bboxes = [
            (xmin * original_size[0], ymin * original_size[1], xmax * original_size[0], ymax * original_size[1])
            for (xmin, ymin, xmax, ymax) in adjusted_pred_bboxes
        ]


    gt_bboxes.append(boxes)
    pred_bboxes.append(adjusted_pred_bboxes)

# Prepare the results for saving
results = []

# Iterate through ground truth and predicted bounding boxes
for gt, pred in zip(gt_bboxes, pred_bboxes):
    # Convert ground truth and predicted bounding boxes into the required format
    gt_str = str(gt)  # Convert list of tuples to string
    pred_str = str(pred)  # Convert list of tuples to string
    
    # Append each row as a dictionary
    results.append({
        'gt_bboxes': gt_str,
        'pred_bboxes': pred_str
    })

# Convert the results into a DataFrame
df = pd.DataFrame(results)

# Save to CSV
output_csv = "../res/test_maira2_raw.csv"  # Update with desired path
df.to_csv(output_csv, index=False)

print(f"Results saved to {output_csv}")




