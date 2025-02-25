import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import ast
import random
import yaml
from transformers import AutoModelForCausalLM, AutoProcessor
import numpy as np
import supervision as sv
import os
import albumentations as A
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from Florence2.preprocessor_florence2 import Florence2Processor  



def normalize_coordinates(bbox, image_shape, scale_factor=1000):
    if bbox is None or any(coord is None for coord in bbox):
        print('bbox:', bbox)
        return "<loc_0><loc_0><loc_0><loc_0>"
    x1, y1, x2, y2 = bbox
    # import pdb; pdb.set_trace() 
    h, w, c = image_shape  # image shape (H, W)
    normalized_x1 = int((x1 / w) * scale_factor)
    normalized_y1 = int((y1 / h) * scale_factor)
    normalized_x2 = int((x2 / w) * scale_factor)
    normalized_y2 = int((y2 / h) * scale_factor)
    return f"<loc_{normalized_x1}><loc_{normalized_y1}><loc_{normalized_x2}><loc_{normalized_y2}>"


import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import cv2

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import cv2

class Padchest_Dataset(Dataset):
    def __init__(self, csv_path, root, transform=None):

        self.data_info = pd.read_csv(csv_path)
        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        # Get the image path and load the image
        img_path = self.root + self.data_info.iloc[index]['img_path']
        img_array = np.array(Image.open(img_path))

        # Normalize the image
        img_array = (img_array / img_array.max()) * 255
        img = Image.fromarray(img_array.astype('uint8')).convert('RGB')

        # Apply transformation if provided
        if self.transform:
            img_np = np.array(img)
            img = self.transform(image=img_np)['image']

        # Get the entire row (annotation) from the DataFrame
        annotation = self.data_info.iloc[index]  # Return the entire row as a Pandas Series

        return {
            "image": img,
            "annotation": annotation  # Return the entire row
        }

    def __len__(self):
        return len(self.data_info)


import yaml

# Load YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

class Inference_anomaly_Detection(Dataset):
    def __init__(self, config, dataset, target_anomaly, target_question):
        # Initialize the dataset
        self.base_dataset = dataset
        
        # Load questions and question2task from the YAML config
        self.question2task = load_yaml(config['prompts_yaml'])
        self.anomaly_definition = load_yaml(config['anomaly_definition'])
        self.target_anomaly = target_anomaly
        self.target_question = target_question

    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        image = sample['image']
        annotation = sample['annotation']
        
        # Define the anomaly and ground truth
        anomaly = self.target_anomaly
        ground_truth = annotation[anomaly]
        
        # Get the question and task
        question_template = self.target_question
        task = self.question2task[question_template]
        
        # Format the question based on the template
        if question_template == 'Detect anomaly {input}.':
            question = question_template.replace('{input}', anomaly)
            answer = anomaly if ground_truth == 1 else f"{anomaly} is not present in the image."
        elif question_template == 'Detect anomaly {input}, defined as "{input}".':
            question = question_template.replace('{input}', anomaly)
            question = question.replace('{input}', self.anomaly_definition[anomaly])
            answer = anomaly if ground_truth == 1 else f"{anomaly} is not present in the image."
        else:
            raise ValueError(f"Invalid question: {question_template}. Expected one of ['Detect anomaly {input}.', 'Detect anomaly {input}, defined as \"{input}\".']")

        return {
            'image': image,
            'question': question,
            'answer': answer, 
            'task': task
        }
    


import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class VindrDataset(Dataset):
    def __init__(self, img_root, annotation_csv, split='train', data_pct=1.0, transform=None):
        self.img_root = img_root
        self.transform = transform
        
        # Check if split is valid
        if split not in ['train', 'test', 'validate']:
            raise ValueError(f"Invalid split: {split}. Expected one of ['train', 'test', 'validate'].")
        
        # Check if data_pct is valid
        if not (0 < data_pct <= 1):
            raise ValueError(f"data_pct should be in the range (0, 1], got {data_pct}")
        
        self.annotations = pd.read_csv(annotation_csv)
        if split == 'train':
            self.annotations = self.annotations[(self.annotations['split'] == 'train') | (self.annotations['split'] == 'validate')].reset_index(drop=True)
        else:
            self.annotations = self.annotations[self.annotations['split'] == split].reset_index(drop=True)

        if self.annotations.empty:
            raise ValueError(f"No data available for split: {split}")

        # Sample data based on data_pct
        if data_pct < 1.0:
            sampled_indices = np.random.choice(len(self.annotations), size=int(len(self.annotations) * data_pct), replace=False)
            self.annotations = self.annotations.iloc[sampled_indices].reset_index(drop=True)
        print(f"Loaded {len(self.annotations)} samples for split: {split} with data_pct: {data_pct}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load image
        img_id = self.annotations.iloc[idx]['image_id']
        img_path = os.path.join(self.img_root, f"{img_id}.png")  # Assuming .jpg format
        image = np.array(Image.open(img_path).convert("RGB"))
        
        
        class_name = self.annotations.iloc[idx]['class_name'].replace('vindrcxr/', '')
        if class_name == 'Nodule/Mass':
            class_name = 'Nodule or Mass'
        class_name = class_name.lower()
        boxes = ast.literal_eval(self.annotations.iloc[idx]['bboxes'])
        # random.shuffle(boxes) # incase seveal bbox and rembmer ber the box order.

        return {
            'image': image,  # Convert image to tensor
            'boxes': boxes,  # Keep boxes as list
            'label': class_name  # Keep labels as list of strings
        }
    


class VindrDataset_unkown(Dataset):
    def __init__(self, img_root, annotation_csv, split='train', data_pct=1.0, transform=None):
        self.img_root = img_root
        self.transform = transform
        self.healthy_list = pd.read_csv('/u/home/lj0/Code/florence2/preprocess/vindr/unique_no_finding_image_ids.csv')
        
        # Check if split is valid
        if split not in ['train', 'test', 'validate']:
            raise ValueError(f"Invalid split: {split}. Expected one of ['train', 'test', 'validate'].")
        
        # Check if data_pct is valid
        if not (0 < data_pct <= 1):
            raise ValueError(f"data_pct should be in the range (0, 1], got {data_pct}")
        
        self.annotations = pd.read_csv(annotation_csv)
        if split == 'train':
            self.annotations = self.annotations[(self.annotations['split'] == 'train') | (self.annotations['split'] == 'validate')].reset_index(drop=True)
        else:
            self.annotations = self.annotations[self.annotations['split'] == split].reset_index(drop=True)

        if self.annotations.empty:
            raise ValueError(f"No data available for split: {split}")

        # Sample data based on data_pct
        if data_pct < 1.0:
            sampled_indices = np.random.choice(len(self.annotations), size=int(len(self.annotations) * data_pct), replace=False)
            self.annotations = self.annotations.iloc[sampled_indices].reset_index(drop=True)
        print(f"Loaded {len(self.annotations)} samples for split: {split} with data_pct: {data_pct}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load image
        img_id = self.annotations.iloc[idx]['image_id']
        img_path = os.path.join(self.img_root, f"{img_id}.png")  # Assuming .jpg format
        class_name = self.annotations.iloc[idx]['class_name'].replace('vindrcxr/', '')
        if class_name == 'Nodule/Mass':
            class_name = 'Nodule or Mass'
        class_name = class_name.lower()
        image = np.array(Image.open(img_path).convert("RGB")) 
        boxes = ast.literal_eval(self.annotations.iloc[idx]['bboxes'])
        # random.shuffle(boxes) # incase seveal bbox and rembmer ber the box order.
        # Add positive samples with a 20% chance
        # if random.random() < 0.2:
        #     if idx < len(self.healthy_list):
        #         img_id = self.healthy_list.iloc[idx]['image_id']
        #         img_path = os.path.join(self.img_root, f"{img_id}.png")
        #     else:
        #         img_id = random.choice(self.healthy_list['image_id'])
        #         img_path = os.path.join(self.img_root, f"{img_id}.png")
            
        #     # Update the image with a healthy sample
        #     image = np.array(Image.open(img_path).convert("RGB"))
        #     class_name = 'healthy'  # Label this as a healthy class

        return {
            'image': image,  # Convert image to tensor
            'boxes': boxes,  # Keep boxes as list
            'label': class_name  # Keep labels as list of strings
        }

# Common Transformations for Detection
def get_transform(train=True):
    if train:
        return A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(512, 512),  # Ensure output size is 512x512
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Resize(512, 512),  # Ensure output size is 512x512
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


class DetInstructDataset_vindr(Dataset):
    # def __init__(self, base_dataset, scale_factor=1000, task="<OPEN_VOCABULARY_DETECTION>",task_prompt="Locate {input} in the image.", max_classes=5, max_invalid_cls=2):
    def __init__(self, base_dataset, task="<CAPTION_TO_PHRASE_GROUNDING>",task_prompt="Locate the phrases in the caption: {input}.", use_definition=True):
        self.base_dataset = base_dataset
        self.task_prompt = task_prompt
        self.task = task
        self.scale_factor = 1000 
        self.definition = yaml.safe_load(open('/u/home/lj0/Code/florence2/preprocess/vindr/definition_shorter.yaml'))
        # self.definition = yaml.safe_load(open('/u/home/lj0/Code/florence2/preprocess/vindr/definition_orignial.yaml'))
        self.use_definition = use_definition
        print('❗ Use definition:', self.use_definition)


    def __len__(self):
        return len(self.base_dataset)

    def normalize_coordinates(self, bbox, image_shape):
        x1, y1, x2, y2 = bbox
        h, w = image_shape[:2] # image shape (H, W, C)
        normalized_x1 = int((x1 / w) * self.scale_factor)
        normalized_y1 = int((y1 / h) * self.scale_factor)
        normalized_x2 = int((x2 / w) * self.scale_factor)
        normalized_y2 = int((y2 / h) * self.scale_factor)
        return f"<loc_{normalized_x1}><loc_{normalized_y1}><loc_{normalized_x2}><loc_{normalized_y2}>"

    def __getitem__(self, idx):
        # Get data from the base dataset
        sample = self.base_dataset[idx]
        image = sample['image']
        bounding_boxes = sample['boxes']
        det_obj = sample['label']
        definition = self.definition[det_obj]
        answer = []
        for bbox in bounding_boxes:
            if len(bbox) != 4:
                raise ValueError(f"Bounding box {bbox} has invalid length: {len(bbox)}")
            locs = self.normalize_coordinates(bbox, image.shape)
            answer.append(f"{det_obj}{locs}")
        final_answer = "".join(answer)
        if self.use_definition:
                det_obj = '{} means {}.'.format(det_obj, definition)
        # Generate the task-specific prompt
        task_prompt = self.task_prompt.format(input=det_obj)
        task_prompt = self.task + task_prompt
        

        return {
            'image': image,
            'question': task_prompt,
            'answer': final_answer,
            'task': self.task
        }

import yaml
class DetInstructDataset_vindr_Ukown(Dataset):
    def __init__(self, base_dataset, task="<CAPTION_TO_PHRASE_GROUNDING>",task_prompt="Locate the phrases in the caption: {input}.", use_definition=True):
        self.base_dataset = base_dataset
        self.task_prompt = task_prompt
        self.task = task
        self.scale_factor = 1000  
        self.train_cls = ['aortic enlargement', 'cardiomegaly', 'pleural thickening', 'pulmonary fibrosis', 'lung opacity', 'other lesion', 'pleural effusion', 'nodule or mass', 'calcification', 'ild', 'consolidation', 'atelectasis', 'enlarged pa', 'rib fracture', 'lung cavity', 'clavicle fracture']
        self.test_cls = ['infiltration', 'mediastinal shift', 'pneumothorax', 'emphysema', 'lung cyst', 'edema']
        self.definition = yaml.safe_load(open('../configs/vindr_definition.yaml'))
        self.use_definition = use_definition



    def __len__(self):
        return len(self.base_dataset)

    def normalize_coordinates(self, bbox, image_shape):
        x1, y1, x2, y2 = bbox
        h, w = image_shape[:2] # image shape (H, W, C)
        normalized_x1 = int((x1 / w) * self.scale_factor)
        normalized_y1 = int((y1 / h) * self.scale_factor)
        normalized_x2 = int((x2 / w) * self.scale_factor)
        normalized_y2 = int((y2 / h) * self.scale_factor)
        return f"<loc_{normalized_x1}><loc_{normalized_y1}><loc_{normalized_x2}><loc_{normalized_y2}>"

    def __getitem__(self, idx):
        # Get data from the base dataset
        sample = self.base_dataset[idx]
        image = sample['image']
        bounding_boxes = sample['boxes']
        det_obj = sample['label']
        definition = self.definition[det_obj]
        if det_obj != 'healthy':
            answer = []
            for bbox in bounding_boxes:
                if len(bbox) != 4:
                    raise ValueError(f"Bounding box {bbox} has invalid length: {len(bbox)}")
                locs = self.normalize_coordinates(bbox, image.shape)
                answer.append(f"{det_obj}{locs}")
            final_answer = "".join(answer)
            # Generate the task-specific prompt
            if self.use_definition:
                det_obj = '{} means {}.'.format(det_obj, definition)
            task_prompt = self.task_prompt.format(input=det_obj)
            task_prompt = self.task + task_prompt
        else:
            det_obj = self.train_cls.random.choice()
            final_answer = 'No finding of {}'.format(det_obj)
            task_prompt = self.task_prompt.format(input=det_obj)
            task_prompt = self.task + task_prompt

        return {
            'image': image,
            'question': task_prompt,
            'answer': final_answer,
            'task': self.task
        }

