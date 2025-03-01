

import io
import os
import json
import torch
import html
import base64
import numpy as np
from PIL import Image
import wandb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)
from peft import LoraConfig, get_peft_model
import supervision as sv
import re

# Define the render inline and example functions for visualization
def render_inline(image: Image.Image, resize=(128, 128)):
    """Convert image into inline html."""
    image.resize(resize)
    with io.BytesIO() as buffer:
        image.save(buffer, format='jpeg')
        image_b64 = str(base64.b64encode(buffer.getvalue()), "utf-8")
        return f"data:image/jpeg;base64,{image_b64}"

def render_example(image: Image.Image, response):
    try:
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)
        image = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(image.copy(), detections)
        image = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(image, detections)
    except:
        print('failed to render model response')
    return f"""
<div style="display: inline-flex; align-items: center; justify-content: center;">
    <img style="width:256px; height:256px;" src="{render_inline(image, resize=(128, 128))}" />
    <p style="width:512px; margin:10px; font-size:small;">{html.escape(json.dumps(response))}</p>
</div>
"""




def parse_labels_and_boxes(input_text, image_width, image_height):
    pattern = r"(\w+(?: \w+)*)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>"
    matches = re.findall(pattern, input_text)

    labels_and_boxes = []

    for match in matches:
        label = match[0] 
        normalized_box = np.array([int(match[1]), int(match[2]), int(match[3]), int(match[4])])

        box = normalized_box / np.array([1000, 1000, 1000, 1000])  
        box = box * np.array([image_width, image_height, image_width, image_height])

        labels_and_boxes.append({"label": label, "box": box.tolist()})

    return labels_and_boxes



def convert_to_od_format(input_data):
    # Initialize the output structure
    output_data = {
        '<OD>': {
            'bboxes': [],
            'labels': []
        }
    }

    # Map input data to the desired structure
    for item in input_data:
        bbox = [item['box'][0], item['box'][1], item['box'][2], item['box'][3]]
        label = item['label']

        output_data['<OD>']['bboxes'].append(bbox)
        output_data['<OD>']['labels'].append(label)

    return output_data




from PIL import Image, ImageDraw, ImageFont

def combine_images(image1, image2):
    """
    Combines two images side by side into a single canvas with captions "left: GT" and "right: Prediction".
    """
    width, height = image1.size
    combined_width = width * 2
    combined_image = Image.new("RGB", (combined_width, height))

    # Paste the images
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width, 0))

    # Draw text (left: GT, right: Prediction)
    draw = ImageDraw.Draw(combined_image)

    # Define font and size (default to a simple font if PIL's default is not available)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Add text to the image (left and right captions)
    text_left = "left: GT"
    text_right = "right: Prediction"

    # Add text to the left and right sections of the combined image
    draw.text((10, 10), text_left, font=font, fill="white")
    draw.text((width + 10, 10), text_right, font=font, fill="white")

    return combined_image




def render_inference_results2(model, processor, batch, DEVICE,sample_size: int):
    """
    Renders inference results for a single batch of data during validation.

    Args:
        model: The model used for predictions.
        processor: The processor used for encoding and decoding.
        batch: A batch from the dataloader containing images, questions, and answers.
        DEVICE: The device to run inference on.

    Returns:
        list: List of combined images with ground truth and predictions.
        list: List of captions describing the predictions and ground truths.
    """
    images_to_log = []
    captions = []

    # Unpack the batch
    # questions = batch['question']  # List of questions/prompts
    # answers = batch['answer']  # List of ground truth answers
    # images = batch['image']  # Batch of images (as tensors or PIL images)
    images,questions, answers, tasks = batch
    bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    for i in range(min(len(images), sample_size)):
        # Get individual components
        image = images[i]
        question = questions[i]
        ground_truth = answers[i]
        task = tasks[i]
        if task == '<OD1>':task = '<OD>'
        if task == '<OD2>':task = '<OD>'

        # Annotate predictions
        

        # Parse ground truth and convert to object detection format
        
        ground_truth_boxes = parse_labels_and_boxes(ground_truth, image.shape[0], image.shape[1])
        ground_truth_od = convert_to_od_format(ground_truth_boxes)
        # ground_truth_od = processor.post_process_generation(
        #         ground_truth, 
        #         task=task, 
        #         image_size=image.shape[:2]
        #     )
        gt = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, ground_truth_od, resolution_wh=image.shape[:2])

        # Annotate ground truth on the image
        image_with_ground_truth = bounding_box_annotator.annotate(image.copy(), gt)
        image_with_ground_truth = label_annotator.annotate(image_with_ground_truth, gt)
        # import pdb; pdb.set_trace()

        # # Process inputs for the model
        # if DEVICE.type=='CUDA':
        #     inputs = processor(text=question, images=image, return_tensors="pt").to(DEVICE)
        # else:
        #     inputs = processor(text=question, images=image, return_tensors="pt")
        inputs = processor(text=question, images=image, return_tensors="pt").to(DEVICE)

        # Generate predictions
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=300,  # Maximum number of tokens to generate 
            num_beams=3
        )

        # generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=1024,  num_beams=3)
        # import pdb; pdb.set_trace()
        # Decode and post-process predictions
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        predicted_boxes = processor.post_process_generation(
            generated_text,
            task=task,
            image_size=image.shape[:2]
        )

        prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, predicted_boxes, resolution_wh=image.shape)

        # Annotate predictions on the image
        image_with_predictions = bounding_box_annotator.annotate(image.copy(), prediction)
        image_with_predictions = label_annotator.annotate(image_with_predictions, prediction)

        # Combine ground truth and predictions for visualization
        image_with_ground_truth = Image.fromarray(image_with_ground_truth.astype(np.uint8))
        image_with_predictions = Image.fromarray(image_with_predictions.astype(np.uint8))
        combined_image = combine_images(image_with_ground_truth, image_with_predictions)

        # Construct captions with both ground truth and predicted text
        caption = f"Ground truth: {question} -> {ground_truth}\nPredicted: {predicted_boxes}\nPredicted-text: {generated_text}"
        
        # Append the image and caption to the results
        images_to_log.append(combined_image)
        captions.append(caption)

    return images_to_log, captions


from difflib import get_close_matches
def correct_class_names(predicted_names, valid_classes, threshold=0.8):
    corrected_names = []
    for name in predicted_names:
        matches = get_close_matches(name, valid_classes, n=1, cutoff=threshold)
        corrected_names.append(matches[0] if matches else name) 
    return corrected_names


import numpy as np
from PIL import Image
# import sv




# CLASSES = ['Lung Opacity', 'Infiltration', 'Consolidation', 'Nodule or Mass', 'Pleural thickening', 
#            'Aortic enlargement', 'Pulmonary fibrosis', 'ILD', 'Cardiomegaly', 'Other lesion', 
#            'Pleural effusion', 'Calcification', 'Enlarged PA', 'Lung cavity', 'Atelectasis', 
#            'Mediastinal shift', 'Lung cyst', 'Pneumothorax', 'Emphysema', 'Clavicle fracture', 
#            'Rib fracture', 'Edema']
CLASSES = ['Pleural thickening', 'Aortic enlargement', 'Pulmonary fibrosis', 'Cardiomegaly', 'Nodule or Mass', 'Lung Opacity', 'Other lesion', 'Pleural effusion', 'ILD', 'Infiltration', 'Calcification', 'Consolidation', 'Atelectasis', 'Rib fracture', 'Mediastinal shift', 'Enlarged PA', 'Pneumothorax', 'Emphysema', 'Lung cavity', 'Lung cyst', 'Clavicle fracture', 'Edema']
CLASSES = [cls.lower() for cls in CLASSES]


from difflib import get_close_matches

def correct_class_names(predicted_names, valid_classes, threshold=0.8):
    corrected_names = []
    for name in predicted_names:
        matches = get_close_matches(name, valid_classes, n=1, cutoff=threshold)
        corrected_names.append(matches[0] if matches else name) 
    return corrected_names

from difflib import get_close_matches
def evluate_resultsevluate_results(model, inputs,processor, answers, images,batch_idx, questions):
    # import pdb;pdb.set_trace()
    bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    color_annotator = sv.ColorAnnotator(color_lookup=sv.ColorLookup.INDEX)
    
    generated_ids = model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=300,num_beams=1)
    
    # model.generate(input_ids=inputs["input_ids"],pixel_values=inputs["pixel_values"],max_new_tokens=300,  num_beams=3)
    
    # Decode the predicted answers
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)

    targets = []
    predictions = []
    img_list = []
    pred_text_list = []
    gt_text_list = []
    pred_label = []
    gt_label = []
    captions = []
    for i, text in enumerate(generated_text):
        answer = processor.post_process_generation(text, task='<CAPTION_TO_PHRASE_GROUNDING>', image_size=images[i].shape[:2])
        gt_answer = processor.post_process_generation(answers[i], task='<CAPTION_TO_PHRASE_GROUNDING>', image_size=images[i].shape[:2])
        
        # Create detections for both the ground truth and predicted answers
        # import pdb;pdb.set_trace()
        gt = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, gt_answer, resolution_wh=images[i].shape)
        gt.class_id = np.array([CLASSES.index(class_name) for class_name in gt['class_name']])
        # import pdb;pdb.set_trace()
        class_label = gt['class_name'][0]
        image_with_ground_truth = bounding_box_annotator.annotate(images[i].copy(), gt)
        image_with_ground_truth = label_annotator.annotate(image_with_ground_truth, gt)
        pred_text_list.append(answer)
        gt_text_list.append(gt_answer)
        # if gt['class_name'][0] == 'infiltration':
        #     import pdb;pdb.set_trace()
        
        prediction = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, answer, resolution_wh=images[i].shape)
        if prediction['class_name'] is not None and len(prediction['class_name']) > 0:
            prediction = prediction[~np.char.startswith(prediction['class_name'], 'mark')] 
            
        
            corrected_class_names = []
            # import pdb;pdb.set_trace()
            for class_name in prediction['class_name']:
                matches = get_close_matches(class_name, CLASSES, n=1, cutoff=0.5) 
                corrected_class_names.append(matches[0] if matches else class_name)
            prediction['class_name'] = corrected_class_names 
            prediction = prediction[np.isin(prediction['class_name'], CLASSES)] 
            
            prediction.class_id = np.array([CLASSES.index(class_name) for class_name in prediction['class_name']])
            prediction.confidence = np.ones(len(prediction))
            # import pdb;pdb.set_trace()
           
            per_sample_pred_cls = prediction['class_name'].tolist()
            per_sample_gt_cls = gt['class_name'].tolist()
            pred_label.append(per_sample_pred_cls)
            gt_label.append(per_sample_gt_cls)

            targets.append(gt)
            predictions.append(prediction)
            if i<10:
                prediction['class_name'] = ['pred_'+ class_label]*len(prediction['class_name']) 
                image_with_predictions = bounding_box_annotator.annotate(image_with_ground_truth.copy(), prediction)
                image_with_predictions = color_annotator.annotate(image_with_predictions, prediction)
                # image_with_predictions = label_annotator.annotate(image_with_predictions, prediction)
                # image_with_ground_truth = Image.fromarray(image_with_ground_truth.astype(np.uint8))
                image_with_predictions = Image.fromarray(image_with_predictions.astype(np.uint8))
                # res = combine_images(image_with_ground_truth, image_with_predictions)
                # img_list.append(res)
                img_list.append(image_with_predictions)
                captions.append(f"Question: {questions[i]}\nGround truth: {gt_answer}\nPredicted: {answer}")
        else:
            # import pdb;pdb.set_trace()
            # prediction.class_id = np.array([-1])
            targets.append(gt)
            predictions.append(prediction)
            if i<10:
                img_list.append(image_with_ground_truth)
                captions.append(f"Question: {questions[i]}\nGround truth: {gt_answer}\nPredicted: {answer}")
            pred_label.append([])
            gt_label.append(gt['class_name'])

    return {
        "res_samples": img_list,  
        "predictions": predictions, 
        "targets": targets,        
        "text_pred_answer":pred_text_list, 
        "text_gt_answer":gt_text_list,
        "pred_label":pred_label,
        "gt_label":gt_label,
        "captions":captions
    }
    # mean_average_precision = sv.MeanAveragePrecision.from_detections(predictions=predictions,targets=targets)



        # print(confusion_matrix.matrix)

    #     # Ensure valid class names
    #     prediction['class_name'] = correct_class_names(prediction['class_name'], CLASSES)
    #     prediction = prediction[np.isin(prediction['class_name'], CLASSES)]

    #     # Assign class_id and confidence for the prediction
    #     prediction.class_id = np.array([CLASSES.index(class_name) for class_name in prediction['class_name']])
    #     prediction.confidence = np.ones(len(prediction))

    #     # Annotate images with ground truth and predictions
    #     image_with_predictions = annotate_image(images[i], prediction, bounding_box_annotator, label_annotator)
    #     image_with_ground_truth = annotate_image(images[i], gt, bounding_box_annotator, label_annotator)

    #     # Convert to PIL images for saving
    #     image_with_ground_truth = Image.fromarray(image_with_ground_truth.astype(np.uint8))
    #     image_with_predictions = Image.fromarray(image_with_predictions.astype(np.uint8))

    #     # Combine and save images
    #     combined_image = combine_images(image_with_ground_truth, image_with_predictions)
    #     combined_image.save(f"./combined_image_{batch_idx}_{i}.png")

    #     processed_predictions.append((image_with_ground_truth, image_with_predictions))  # You could also store metrics here

    # return processed_predictions
    # return 'ok'



def annotate_image(image, detections, bounding_box_annotator, label_annotator):
    """
    Annotate an image with bounding boxes and labels.
    
    Args:
        image (np.array): The image to annotate.
        detections (sv.Detections): The detections object containing bounding box information.
        bounding_box_annotator (object): Annotator for bounding boxes.
        label_annotator (object): Annotator for labels.
    
    Returns:
        np.array: The annotated image.
    """
    image_with_predictions = bounding_box_annotator.annotate(image.copy(), detections)
    image_with_predictions = label_annotator.annotate(image_with_predictions, detections)
    return image_with_predictions
