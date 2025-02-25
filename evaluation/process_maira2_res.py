import pandas as pd
from tqdm import tqdm
import ast

CLASSES = [
    'Pleural thickening', 'Aortic enlargement', 'Pulmonary fibrosis', 'Cardiomegaly', 
    'Nodule or Mass', 'Lung Opacity', 'Other lesion', 'Pleural effusion', 'ILD', 
    'Infiltration', 'Calcification', 'Consolidation', 'Atelectasis', 'Rib fracture', 
    'Mediastinal shift', 'Enlarged PA', 'Pneumothorax', 'Emphysema', 'Lung cavity', 
    'Lung cyst', 'Clavicle fracture', 'Edema'
]
CLASSES = [cls.lower() for cls in CLASSES]

dataset = pd.read_csv('../annotations/vindr_dataset.csv')
test_dataset = dataset[dataset['split'] == 'test'].reset_index(drop=True)

result = pd.read_csv('../res/test_maira2_raw.csv')

required_columns = ['tgt_bbox', 'pred_bbox', 'tgt_class_id', 'pred_class_id', 'tgt_class_name']
for col in required_columns:
    if col not in result.columns:
        result[col] = None  
for idx, row in tqdm(result.iterrows(), total=len(result), desc="Processing results"):
    class_name = test_dataset.loc[idx, 'class_name'].replace('vindrcxr/', '')
    if class_name.lower() == 'nodule/mass':
        class_name = 'Nodule or Mass'
    
    class_name = class_name.lower()
    
    if class_name not in CLASSES:
        print(f"Warning: '{class_name}' not found in CLASSES list. Skipping index {idx}.")
        continue

    class_id = CLASSES.index(class_name)

    pred_bboxes = ast.literal_eval(result.loc[idx, 'pred_bboxes'])
    gt_bboxes = ast.literal_eval(result.loc[idx, 'gt_bboxes'])

    result.at[idx, 'tgt_bbox'] = gt_bboxes
    result.at[idx, 'pred_bbox'] = pred_bboxes
    result.at[idx, 'tgt_class_id'] = [class_id] * len(gt_bboxes)
    result.at[idx, 'pred_class_id'] = [class_id] * len(pred_bboxes)
    result.at[idx, 'tgt_class_name'] = [class_name] * len(gt_bboxes)
    result.at[idx, 'pred_class_name'] = [class_name] * len(pred_bboxes)

output_csv = "../res/maira_vindr_res.csv"
result.to_csv(output_csv, index=False)

print(f"✅ Updated test dataset saved to {output_csv}")