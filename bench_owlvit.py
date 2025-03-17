import roboflow
import torch
from roboflow.core.project import Project
from tqdm import tqdm
from typing import List
import os
import shutil
import numpy as np
from inference.models.owlv2.owlv2 import OwlV2 
from inference.core.entities.requests.owlv2 import OwlV2InferenceRequest
import supervision as sv
import json
from copy import deepcopy
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from transformers import pipeline

api_key = ""
rf = roboflow.Roboflow(api_key=api_key)
data_dir = os.path.join("scratch", "owlvit_data")
save_dir = os.path.join("scratch", "owlvit_results")
os.makedirs(save_dir, exist_ok=True)

import supervision as sv
import json
from pathlib import Path
import numpy as np

API_KEY_OLD = ""
rf_old = roboflow.Roboflow(api_key=API_KEY_OLD)
workspace = rf_old.workspace("roboflow100vl-fsod")
# full_api_key = "jyOhx1yGm4JczykPv075"
fsod_api_key = "cUalZjFpte9g6QpLKlTJ"
rf_fosd = roboflow.Roboflow(api_key=fsod_api_key)
workspace = rf_fosd.workspace("rf-100-vl-fsod")

# Create a class to handle COCO format export
class COCOExporter:
    def __init__(self, categories, output_path="annotations.json"):
        self.output_path = os.path.join(save_dir, output_path)
        self.images = []
        self.annotations = []
        self.categories = [{"id": i, "name": name} for i, name in enumerate(categories)]
        self.image_id = 0
        self.annotation_id = 0
    
    def add_image(self, image_path, detections):
        # Get image info

        image = Image.open(image_path)
        image_info = {
            "id": self.image_id,
            "file_name": Path(image_path).name,
            "width": image.width,
            "height": image.height
        }
        self.images.append(image_info)
        
        # Process detections
        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            # Convert to COCO format (x, y, width, height)
            x, y, x2, y2 = xyxy
            width = x2 - x
            height = y2 - y
            bbox = [float(x), float(y), float(width), float(height)]
            
            # Create annotation
            annotation = {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": int(detections.class_id[i]),
                "bbox": bbox,
                "area": float(width * height),
                "iscrowd": 0,
                "segmentation": []  # Empty for box-only annotations
            }
            
            if hasattr(detections, "confidence") and detections.confidence is not None:
                annotation["score"] = float(detections.confidence[i])
                
            self.annotations.append(annotation)
            self.annotation_id += 1
        
        self.image_id += 1
    
    def save(self):
        coco_output = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(coco_output, f)
        
        print(f"COCO annotations saved to {self.output_path}")

# Example usage
def save_detections_as_coco(image_paths, all_detections, categories, output_path="coco_annotations.json"):
    exporter = COCOExporter(categories=categories, output_path=output_path)
    
    for image_path, detections in zip(image_paths, all_detections):
        exporter.add_image(image_path, detections)
    
    exporter.save()

def download_projects():
    projects_array: List[Project] = []
    for a_project in workspace.project_list:
        proj = Project(api_key, a_project, workspace.model_format)
        projects_array.append(proj)

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    for project in tqdm(projects_array):
        last_version = project.versions()[-1]
        print(project.name)
        download_dir = os.path.join(data_dir, project.name)
        downloaded = last_version.download(model_format="coco", location=download_dir)
        for split in ["train", "test"]:
            coco_annotations = os.path.join(download_dir, split, "_annotations.coco.json")
            with open(coco_annotations, "r") as f:
                coco_data = json.load(f)
            coco_data = clean_coco_annotations(coco_data)
            with open(coco_annotations, "w") as f:
                json.dump(coco_data, f)

def clean_coco_annotations(data_ann):
    new_data_ann = {}
    if data_ann['info']:
        new_data_ann['info'] = data_ann['info']
    if data_ann['licenses']:
        new_data_ann['licenses'] = data_ann['licenses']

    # confirm if category 0 is none
    assert(data_ann['categories'][0]['supercategory']=='none'), "Need to change logic for removing category 0 from dataset in preprocessing"

    # data_ann['categories'] = [cat for cat in data_ann['categories'] if cat['id']!=0]
    new_data_ann['categories'] = [{'id': cat['id']-1, 'name': cat['name'], 'supercategory': cat['supercategory']} for cat in data_ann['categories'] if cat['id']!=0]

    new_data_ann['images'] = data_ann['images']
    new_data_ann['annotations'] = deepcopy(data_ann['annotations'])

    for ann in new_data_ann['annotations']:
        ann['category_id'] = ann['category_id']-1
    return new_data_ann

def get_category_map(data_ann):
    return {cat["name"]: cat["id"] for cat in data_ann["categories"]}

def benchmark_owlvit():
    projects = os.listdir(data_dir)
    all_stats = {}
    for project in tqdm(projects, desc="Benchmarking projects"):
        stats = benchmark_project(project)
        all_stats[project] = stats
        with open(os.path.join(save_dir, "owlvit_stats_zshot.json"), "w") as f:
            json.dump(all_stats, f)

    return all_stats

def do_prediction(model, image_name, train_dataset, classes, class_map, text=False):
    owl_vit_request = construct_prompt(image_name, train_dataset, classes, text=text)
    response = model.infer_from_request(owl_vit_request)
    for pred in response.predictions:
        pred.class_id = class_map[pred.class_name]
    detections  = sv.Detections.from_inference(response)
    return detections

def do_prediction_label(model, processor, image_name, classes, class_map):
    image = Image.open(image_name).convert("RGB")
    text_queries = classes
    xyxy = []
    confidence = []
    class_id = []
    inputs = processor(images=image, text=[text_queries], return_tensors="pt")
    target_sizes = torch.tensor([image.size[::-1]])
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.1)[0]
    for score, label, box in zip(predictions["scores"], predictions["labels"], predictions["boxes"]):
        xyxy.append(
            np.array([box[0], box[1], box[2], box[3]])
        )
        confidence.append(score)
        class_id.append(label)
    
    if not xyxy:
        return sv.Detections.empty()

    detections = sv.Detections(
        xyxy=np.array(xyxy),
        confidence=np.array(confidence),
        class_id=np.array(class_id),
    )
    return detections

def benchmark_project(project):
    model = OwlV2()
    train_dir = os.path.join(data_dir, project, "train")
    test_dir = os.path.join(data_dir, project, "test")
    coco_annotations = os.path.join(train_dir, "_annotations.coco.json")
    with open(coco_annotations, "r") as f:
        train_data_ann = json.load(f)
        class_map = get_category_map(train_data_ann)
    test_annotations = os.path.join(test_dir, "_annotations.coco.json")
    train_dataset = sv.DetectionDataset.from_coco(images_directory_path=train_dir, annotations_path=coco_annotations)
    test_dataset = sv.DetectionDataset.from_coco(images_directory_path=test_dir, annotations_path=test_annotations)
    classes = train_dataset.classes

    all_detections = []
    gt_detections = []
    for image_name, _, annotations in tqdm(test_dataset, desc=f"Processing {project}"):
        gt_detections.append(annotations)
        detections = do_prediction(model, image_name, train_dataset, classes, class_map, text=True)
        all_detections.append(detections)
    
    save_detections_as_coco(test_dataset.images, all_detections, classes, f"preds_{project}_zshot.json")
    
    map = sv.MeanAveragePrecision.from_detections(all_detections, gt_detections)
    map50 = float(map.map50)
    map50_95 = float(map.map50_95)
    return {"map50": map50, "map50_95": map50_95}

    


def construct_prompt(test_image, dataset, classes, text=False):
    training_data = []
    if not text:
        for image_name, _, annotations in dataset:
            boxes = []
            for xyxy, _, _, class_id, _, _ in annotations:
                image_obj = {
                    "type": "file",
                    "value": image_name,
                }
                x, y, w, h = (xyxy[0] + xyxy[2])//2, (xyxy[1] + xyxy[3])//2, xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                x, y, w, h = int(x), int(y), int(w), int(h)
                boxes.append({
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "cls": classes[int(class_id.item())],
                    "negative": False,
                })
            training_data.append({
                "image": image_obj,
                "boxes": boxes,
            })

        test_image = {
            "type": "file",
            "value": test_image,
        }
        request = OwlV2InferenceRequest(
            image=test_image,
            training_data=training_data,
            visualize_predictions=False,
            confidence=0.1,
        )
    else:
        test_image = {
            "type": "file",
            "value": test_image,
        }
        request = OwlV2InferenceRequest(
            image=test_image,
            training_data=classes,
            visualize_predictions=False,
            confidence=0.1,
        )
    return request

def report_stats():
    with open("scratch/owlvit_results/owlvit_stats_zshot.json", "r") as f:
        stats = json.load(f)
    
    print("Project, MAP50, MAP50_95")
    for project, stat in stats.items():
        print(f"{project}, {stat['map50']}, {stat['map50_95']}")
    
    print("Average MAP50, MAP50_95")
    print(f"{np.mean([stat['map50'] for stat in stats.values()]):.5f}, {np.mean([stat['map50_95'] for stat in stats.values()]):.5f}")
    with open("scratch/name_to_label.json", "r") as f:
        name_to_label = json.load(f)
    
    name_to_label_keys = sorted(list(name_to_label.keys()))
    stats_keys = sorted(list(stats.keys()))

    from collections import defaultdict
    cats = defaultdict(list)
    cats_map50 = defaultdict(list)
    new_name_to_label = {}
    for name_to_label_key, stat_key in zip(name_to_label_keys, stats_keys):
        cats[name_to_label[name_to_label_key]].append(stats[stat_key]['map50_95'])
        cats_map50[name_to_label[name_to_label_key]].append(stats[stat_key]['map50'])
        new_name_to_label[stat_key] = name_to_label[name_to_label_key]
    
    for cat, stats in cats.items():
        print(f"{cat} map50_95: {np.mean(stats):.4f}")
        print(f"{cat} map50: {np.mean(cats_map50[cat]):.4f}")

if __name__ == "__main__":
    # download_projects()
    benchmark_owlvit()
    # report_stats()``