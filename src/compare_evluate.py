import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from rodeo import RoDeO
import supervision as sv
from PIL import Image
import wandb
import os
from PIL import Image, ImageDraw, ImageFont

# Define categories
CLASSES = ['Pleural thickening', 'Aortic enlargement', 'Pulmonary fibrosis', 'Cardiomegaly', 'Nodule or Mass', 
           'Lung Opacity', 'Other lesion', 'Pleural effusion', 'ILD', 'Infiltration', 'Calcification', 'Consolidation', 
           'Atelectasis', 'Rib fracture', 'Mediastinal shift', 'Enlarged PA', 'Pneumothorax', 'Emphysema', 'Lung cavity', 
           'Lung cyst', 'Clavicle fracture', 'Edema']
CLASSES = [cls.lower() for cls in CLASSES]

# Initialize wandb
wandb.init(project="model_comparison", name="Bounding_Box_Visualization")

# Define the visualization function
def visualize_and_upload(row, image_id):
    
    image_path = os.path.join('/home/jun/datasets/vindr/images_512/images_512', f'{image_id}.png')
    row['tgt_class_id'] = ast.literal_eval(row['tgt_class_id'])
    row['pred_class_id'] = ast.literal_eval(row['pred_class_id'])
    row['pred_bbox'] = ast.literal_eval(row['pred_bbox'])
    row['tgt_bbox'] = ast.literal_eval(row['tgt_bbox'])
    tgt_class_name = ast.literal_eval(row["tgt_class_name"])
    pred_bboxes = np.array(row['pred_bbox'])
    tgt_bboxes = np.array(row['tgt_bbox'])

    if pred_bboxes.size == 0:
        pred_bboxes = np.array([[0, 0, 0, 0, row['tgt_class_id'][0]]])
    else:
        pred_bboxes = np.array([[x1, y1, x2 - x1, y2 - y1, row['pred_class_id'][0]] for x1, y1, x2, y2 in pred_bboxes])
    
    tgt_bboxes = np.array([[x1, y1, x2 - x1, y2 - y1, row['tgt_class_id'][0]] for x1, y1, x2, y2 in tgt_bboxes])
    
    # Creating detections using Supervision
    sv_gt = sv.Detections(
        xyxy=np.array(np.array(row['tgt_bbox'])),
        mask=None,
        confidence=None,
        class_id=np.array(row['tgt_class_id']),
        tracker_id=None,
    )

    sv_gt['class_name'] = [tgt_class_name[0]] * len(row['tgt_bbox'])
    
    sv_pred = sv.Detections(
        xyxy=np.array(row['pred_bbox']) if len(row['pred_bbox']) > 0 else np.empty((0, 4)),
        mask=None,
        confidence=np.ones(len(row['pred_bbox'])) if len(row['pred_bbox']) > 0 else np.empty(0),
        class_id=np.array(row['pred_class_id']) if len(row['pred_class_id']) > 0 else np.empty(0),
        tracker_id=None,
    )
    image = Image.open(image_path).convert("RGB")
    # print('image_id:', image_id)
    # import pdb;pdb.set_trace()

    # bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.CLASS)
    # label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.CLASS)
    # color_annotator = sv.ColorAnnotator(color_lookup=sv.ColorLookup.INDEX)
    color_annotator = sv.ColorAnnotator(color_lookup=sv.ColorLookup.CLASS)

    image_with_ground_truth = bounding_box_annotator.annotate(image.copy(), sv_gt)
    image_with_ground_truth = label_annotator.annotate(image_with_ground_truth, sv_gt)
    image_with_predictions = bounding_box_annotator.annotate(image_with_ground_truth.copy(), sv_pred)
    image_with_predictions = color_annotator.annotate(image_with_predictions, sv_pred)
    # # Draw class names on the image
    # draw = ImageDraw.Draw(image_with_predictions)
    # font = ImageFont.load_default()
    # draw.text((10, 10), tgt_class_name[0], fill="red", font=font)

    return image_with_predictions



orignial_dataset = pd.read_csv('../annotations/vindr_dataset.csv')
test_dataset = orignial_dataset[orignial_dataset['split'] == 'test'].reset_index(drop=True)

# Load the CSV files outside the function
csv_files = {
    '../res/maira_vindr_res.csv': 'MAIRA-2',  # MAIRA
    '../res/radVLM_vindr_res.csv': 'RadVLM',  # Knowledge-enhanced
    '../res/our_vindr_res.csv': 'ours',  # RadVLM
}

# Read all CSV files and save to a dictionary
csv_data = {name: pd.read_csv(path) for path, name in csv_files.items()}



total_len = len(test_dataset)
random_idx = np.random.choice(total_len, 10, replace=False)

from matplotlib import pyplot as plt

# import pdb;pdb.set_trace()

for idx in random_idx:
    image_id = test_dataset.loc[idx, 'image_id']
    all_results = []
    for method,csv_df in csv_data.items():
        row = csv_df.iloc[idx]
        pred_img = visualize_and_upload(row, image_id)
        all_results.append([pred_img, method])
    wandb.log({f"Image_{image_id}": [wandb.Image(img, caption=method) for img, method in all_results]})






    