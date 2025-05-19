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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import rf100vl
import traceback

api_key = ""
rf = roboflow.Roboflow(api_key=api_key)
data_dir = os.path.join("scratch", "owlvit_data")
save_dir = os.path.join("scratch", "owlvit_results_new")
os.makedirs(save_dir, exist_ok=True)

import supervision as sv
import json
from pathlib import Path
import numpy as np

API_KEY_OLD = ""
rf_old = roboflow.Roboflow(api_key=API_KEY_OLD)
workspace = rf_old.workspace("roboflow100vl-fsod")
fsod_api_key = ""
rf_fosd = roboflow.Roboflow(api_key=fsod_api_key)
workspace = rf_fosd.workspace("rf20-vl-fsod")

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

LINKS = [
    "https://universe.roboflow.com/rf20-vl-fsod/actions-zzid2-zb1hq-fsod-amih/1",
    "https://universe.roboflow.com/rf20-vl-fsod/aerial-airport-7ap9o-fsod-ddgc/1",
    "https://universe.roboflow.com/rf20-vl-fsod/all-elements-fsod-mebv/1",
    "https://universe.roboflow.com/rf20-vl-fsod/aquarium-combined-fsod-gjvb/2",
    "https://universe.roboflow.com/rf20-vl-fsod/defect-detection-yjplx-fxobh-fsod-amdi/1",
    "https://universe.roboflow.com/rf20-vl-fsod/dentalai-i4clz-fsod-fsuo/1",
    "https://universe.roboflow.com/rf20-vl-fsod/flir-camera-objects-fsod-tdqp/1",
    "https://universe.roboflow.com/rf20-vl-fsod/gwhd2021-fsod-atsv/2",
    "https://universe.roboflow.com/rf20-vl-fsod/lacrosse-object-detection-fsod-uxkt/1",
    "https://universe.roboflow.com/rf20-vl-fsod/new-defects-in-wood-uewd1-fsod-tffp/1",
    "https://universe.roboflow.com/rf20-vl-fsod/orionproducts-vtl2z-fsod-puhv/2",
    "https://universe.roboflow.com/rf20-vl-fsod/paper-parts-fsod-rmrg/1",
    "https://universe.roboflow.com/rf20-vl-fsod/recode-waste-czvmg-fsod-yxsw/1",
    "https://universe.roboflow.com/rf20-vl-fsod/soda-bottles-fsod-haga/1",
    "https://universe.roboflow.com/rf20-vl-fsod/the-dreidel-project-anzyr-fsod-zejm/2",
    "https://universe.roboflow.com/rf20-vl-fsod/trail-camera-fsod-egos/1",
    "https://universe.roboflow.com/rf20-vl-fsod/water-meter-jbktv-7vz5k-fsod-ftoz/1",
    "https://universe.roboflow.com/rf20-vl-fsod/wb-prova-stqnm-fsod-rbvg/2",
    "https://universe.roboflow.com/rf20-vl-fsod/wildfire-smoke-fsod-myxt/1",
    "https://universe.roboflow.com/rf20-vl-fsod/x-ray-id-zfisb-fsod-dyjv/1"
]

def download_projects():
    projects_array: List[Project] = []
    for a_project in workspace.project_list:
        proj = Project(api_key, a_project, workspace.model_format)
        projects_array.append(proj)

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    for project in tqdm(projects_array):
        last_version = max(project.versions(), key=lambda z:  z.version)
        link = [l for l in LINKS if project.name in l][0]
        print(link[-1])
        print(str(last_version))
        assert link[-1] == str(last_version.version)
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
        with open(os.path.join(save_dir, "owlvit_stats_fsod_20.json"), "w") as f:
            json.dump(all_stats, f)

    return all_stats

def do_prediction(model, image_name, train_dataset, classes, class_map, text=False):
    owl_vit_request = construct_prompt(image_name, train_dataset, classes, text=text)
    response = model.infer_from_request(owl_vit_request)
    for pred in response.predictions:
        pred.class_id = class_map[pred.class_name]
    detections  = sv.Detections.from_inference(response)
    image = Image.open(image_name)
    bbox_annotator = sv.BoxAnnotator()
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
        detections = do_prediction(model, image_name, train_dataset, classes, class_map, text=False)
        all_detections.append(detections)
    
    save_detections_as_coco(test_dataset.images, all_detections, classes, f"preds_{project}.json")
    
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
    with open("scratch/owlvit_results_new/owlvit_stats_fsod_20.json", "r") as f:
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

from collections import defaultdict

def format_predictions_for_pycocotools(pred_json_path: str, coco_gt: COCO) -> list:
    """
    Loads predictions from a COCO-style JSON file and formats them for pycocotools evaluation.
    It ensures that image_ids and category_ids in the output list match those in coco_gt.
    """
    try:
        with open(pred_json_path, 'r') as f:
            pred_coco_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {pred_json_path} during formatting.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {pred_json_path} during formatting.")
        return []

    # Ground truth mappings
    gt_filename_to_id = {img['file_name'].split(".rf")[0]: img['id'] for img in coco_gt.dataset.get('images', [])}
    gt_cat_name_to_id = {cat['name']: cat['id'] for cat in coco_gt.dataset.get('categories', [])}

    # Prediction file mappings_
    pred_img_id_to_filename = {img['id']: img['file_name'].split(".rf")[0] for img in pred_coco_data.get('images', [])}
    # The categories in pred_coco_data are from COCOExporter, using train_dataset.classes.
    # IDs are 0-indexed based on the order in train_dataset.classes.
    pred_cat_id_to_name = {cat['id']: cat['name'] for cat in pred_coco_data.get('categories', [])}
    
    pycocotools_preds = []
    for ann in pred_coco_data.get('annotations', []):
        pred_bbox = ann.get('bbox') # [x,y,w,h]
        pred_score = ann.get('score')
        original_pred_img_id = ann.get('image_id')
        original_pred_cat_id = ann.get('category_id')

        if None in [pred_bbox, pred_score, original_pred_img_id, original_pred_cat_id]:
            print(f"Skipping annotation with missing fields in {pred_json_path}")
            continue

        pred_filename = pred_img_id_to_filename.get(original_pred_img_id)
        if not pred_filename:
            print(f"Skipping annotation, image_id {original_pred_img_id} not found in prediction images section.")
            continue
            
        target_gt_image_id = gt_filename_to_id.get(pred_filename)
        if target_gt_image_id is None:
            print(f"Skipping annotation, image filename {pred_filename} not found in ground truth images.")
            exit(0)
            continue

        pred_category_name = pred_cat_id_to_name.get(original_pred_cat_id)
        if not pred_category_name:
            print(f"Skipping annotation, category_id {original_pred_cat_id} not found in prediction categories section.")
            continue

        target_gt_category_id = gt_cat_name_to_id.get(pred_category_name)
        if target_gt_category_id is None:
            print(f"Skipping annotation, category name {pred_category_name} not found in ground truth categories.")
            continue
            
        pycocotools_preds.append({
            "image_id": target_gt_image_id,
            "category_id": target_gt_category_id,
            "bbox": pred_bbox,
            "score": float(pred_score),
        })
    return pycocotools_preds

def summarize(self):
    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
        return mean_s
    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
        stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
        return stats
    def _summarizeKps():
        stats = np.zeros((10,))
        stats[0] = _summarize(1, maxDets=20)
        stats[1] = _summarize(1, maxDets=20, iouThr=.5)
        stats[2] = _summarize(1, maxDets=20, iouThr=.75)
        stats[3] = _summarize(1, maxDets=20, areaRng='medium')
        stats[4] = _summarize(1, maxDets=20, areaRng='large')
        stats[5] = _summarize(0, maxDets=20)
        stats[6] = _summarize(0, maxDets=20, iouThr=.5)
        stats[7] = _summarize(0, maxDets=20, iouThr=.75)
        stats[8] = _summarize(0, maxDets=20, areaRng='medium')
        stats[9] = _summarize(0, maxDets=20, areaRng='large')
        return stats
    if not self.eval:
        raise Exception('Please run accumulate() first')
    iouType = self.params.iouType
    if iouType == 'segm' or iouType == 'bbox':
        summarize = _summarizeDets
    elif iouType == 'keypoints':
        summarize = _summarizeKps
    self.stats = summarize()

def calculate_map_from_saved_predictions():
    projects = os.listdir(data_dir)
    all_recalculated_stats = {}
    EXPECTED_PROJECT_COUNT = 20 
    
    print("Re-calculating mAP from saved predictions using pycocotools...")

    for project in tqdm(projects, desc="Re-evaluating projects"):
        pred_json_path = os.path.join(save_dir, f"preds_{project}.json")
        assert os.path.exists(pred_json_path)
        gt_json_path = os.path.join(data_dir, project, "test", "_annotations.coco.json")
        # test_images_dir is not directly needed for pycocotools mAP calc if jsons are complete

        map50, map50_95 = float('nan'), float('nan') # Initialize

        if not os.path.exists(pred_json_path):
            print(f"Prediction file not found for project {project}, skipping: {pred_json_path}")
            all_recalculated_stats[project] = {"map50": map50, "map50_95": map50_95, "skipped_reason": "no_pred_file"}
            continue

        if not os.path.exists(gt_json_path):
            print(f"Ground truth file not found for project {project}, skipping: {gt_json_path}")
            all_recalculated_stats[project] = {"map50": map50, "map50_95": map50_95, "skipped_reason": "no_gt_file"}
            continue
        
        try:
            coco_gt = COCO(gt_json_path)

            if not coco_gt.dataset.get('images'):
                print(f"Project {project}: Ground truth has no images. Setting mAP to 0.0.")
                map50, map50_95 = 0.0, 0.0
            else:
                predictions_for_eval = format_predictions_for_pycocotools(pred_json_path, coco_gt)
                
                # coco_gt.loadRes can handle an empty list of predictions.
                coco_dt = coco_gt.loadRes(predictions_for_eval)
                
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox') # 'bbox' for object detection
                coco_eval.params.maxDets = [1, 100, 500]
                
                # Optionally restrict evaluation to specific image IDs or category IDs
                # coco_eval.params.imgIds = coco_gt.getImgIds() # Evaluate on all images in GT by default
                # coco_eval.params.catIds = coco_gt.getCatIds() # Evaluate on all GT categories by default

                print(f"\nEvaluating project: {project}")
                coco_eval.evaluate()
                coco_eval.accumulate()
                summarize(coco_eval)
                
                stats = coco_eval.stats
                # stats[0] = AP @ IoU=0.50:0.95 | area=   all | maxDets=100
                # stats[1] = AP @ IoU=0.50      | area=   all | maxDets=100
                map50_95 = stats[0] if stats[0] >= 0 else 0.0 # pycocotools returns -1 if no GT/Preds
                map50 = stats[1] if stats[1] >= 0 else 0.0
                
                # Ensure NaNs (less common with pycocotools if -1 is used, but good practice) become 0.0
                if np.isnan(map50_95): map50_95 = 0.0
                if np.isnan(map50): map50 = 0.0

            all_recalculated_stats[project] = {"map50": map50, "map50_95": map50_95}
            print(rf100vl.util.get_category(rf100vl.util.get_basename(project)))
            # print(f"Project {project} stats: {all_recalculated_stats[project]}") # Optional debug print

        except Exception as e:
            print(f"Error processing project {project}: {e}")
            traceback.print_exc() 
            all_recalculated_stats[project] = {"map50": float('nan'), "map50_95": float('nan'), "skipped_reason": f"error: {str(e)}"}

    # --- Reporting and Assertion Logic (remains largely the same) ---
    print("\nRecalculated Project, MAP50, MAP50_95 (using pycocotools)")
    valid_stats_map50 = []
    valid_stats_map50_95 = []
    
    for project, stat in all_recalculated_stats.items():
        if 'skipped_reason' not in stat and not np.isnan(stat['map50']) and not np.isnan(stat['map50_95']):
            print(f"{project}, {stat['map50']:.5f}, {stat['map50_95']:.5f}")
            valid_stats_map50.append(stat['map50'])
            valid_stats_map50_95.append(stat['map50_95'])
        else:
            reason = stat.get('skipped_reason', 'NaN value')
            print(f"{project}, MAP50: N/A, MAP50_95: N/A (Reason: {reason})")

    print(f"Number of projects for which mAP was successfully calculated: {len(valid_stats_map50)}")
    assert len(valid_stats_map50) == EXPECTED_PROJECT_COUNT, \
        f"Expected {EXPECTED_PROJECT_COUNT} valid mAP50 scores, but got {len(valid_stats_map50)}"
    assert len(valid_stats_map50_95) == EXPECTED_PROJECT_COUNT, \
        f"Expected {EXPECTED_PROJECT_COUNT} valid mAP50-95 scores, but got {len(valid_stats_map50_95)}"
    print(f"Successfully asserted {EXPECTED_PROJECT_COUNT} loaded mAP scores.")

    if valid_stats_map50:
        print(f"Overall Average Recalculated MAP50: {np.mean(valid_stats_map50):.5f}")
    else:
        print("Overall Average Recalculated MAP50: N/A (no valid stats)")
    
    if valid_stats_map50_95:
        print(f"Overall Average Recalculated MAP50_95: {np.mean(valid_stats_map50_95):.5f}")
    else:
        print("Overall Average Recalculated MAP50_95: N/A (no valid stats)")


    cats_recalc_map50 = defaultdict(list)
    cats_recalc_map50_95 = defaultdict(list)

    for project, stat in all_recalculated_stats.items():
        if 'skipped_reason' in stat or np.isnan(stat['map50']) or np.isnan(stat['map50_95']):
            continue 

        label = rf100vl.util.get_category(rf100vl.util.get_basename(project))
        if label:
            cats_recalc_map50[label].append(stat['map50'])
            cats_recalc_map50_95[label].append(stat['map50_95'])
        else:
            print(f"Warning: Project name '{project}' from data_dir not found as a key in name_to_label.json. Skipping for category average.")

    print("\nRecalculated mAP by Dataset Type (using pycocotools):")
    for label in sorted(cats_recalc_map50.keys()): 
        if cats_recalc_map50[label]: 
            avg_map50 = np.mean(cats_recalc_map50[label])
            print(f"{label} Average MAP50: {avg_map50:.4f} (from {len(cats_recalc_map50[label])} projects)")
    for label in sorted(cats_recalc_map50_95.keys()):
        if cats_recalc_map50_95[label]:
            avg_map50_95 = np.mean(cats_recalc_map50_95[label])
            print(f"{label} Average MAP50_95: {avg_map50_95:.4f} (from {len(cats_recalc_map50_95[label])} projects)")
        
    return all_recalculated_stats

import os
import json
import pickle
from collections import defaultdict

def generate_pkl_from_coco_json(coco_json_dir, pkl_output_dir):
    """
    Takes COCO JSON prediction files from coco_json_dir and produces PKL files
    in pkl_output_dir, matching the structure of the original script.
    """

    # NEW_NAMES
# https://universe.roboflow.com/rf20-vl-fsod/actions-zzid2-zb1hq-fsod-amih/1
# https://universe.roboflow.com/rf20-vl-fsod/aerial-airport-7ap9o-fsod-ddgc/1
# https://universe.roboflow.com/rf20-vl-fsod/all-elements-fsod-mebv/1
# https://universe.roboflow.com/rf20-vl-fsod/aquarium-combined-fsod-gjvb/2
# https://universe.roboflow.com/rf20-vl-fsod/defect-detection-yjplx-fxobh-fsod-amdi/1
# https://universe.roboflow.com/rf20-vl-fsod/dentalai-i4clz-fsod-fsuo/1
# https://universe.roboflow.com/rf20-vl-fsod/flir-camera-objects-fsod-tdqp/1
# https://universe.roboflow.com/rf20-vl-fsod/gwhd2021-fsod-atsv/2
# https://universe.roboflow.com/rf20-vl-fsod/lacrosse-object-detection-fsod-uxkt/1
# https://universe.roboflow.com/rf20-vl-fsod/new-defects-in-wood-uewd1-fsod-tffp/1
# https://universe.roboflow.com/rf20-vl-fsod/orionproducts-vtl2z-fsod-puhv/2
# https://universe.roboflow.com/rf20-vl-fsod/paper-parts-fsod-rmrg/1
# https://universe.roboflow.com/rf20-vl-fsod/recode-waste-czvmg-fsod-yxsw/1
# https://universe.roboflow.com/rf20-vl-fsod/soda-bottles-fsod-haga/1
# https://universe.roboflow.com/rf20-vl-fsod/the-dreidel-project-anzyr-fsod-zejm/2
# https://universe.roboflow.com/rf20-vl-fsod/trail-camera-fsod-egos/1
# https://universe.roboflow.com/rf20-vl-fsod/water-meter-jbktv-7vz5k-fsod-ftoz/1
# https://universe.roboflow.com/rf20-vl-fsod/wb-prova-stqnm-fsod-rbvg/2
# https://universe.roboflow.com/rf20-vl-fsod/wildfire-smoke-fsod-myxt/1
# https://universe.roboflow.com/rf20-vl-fsod/x-ray-id-zfisb-fsod-dyjv/1
    pickle_filenames_map = {
        "actions-zzid2-zb1hq-fsod-amih": "actions-zzid2-zb1hq-fsod-amih.pkl",
        "aerial-airport-7ap9o-fsod-ddgc": "aerial-airport-7ap9o-fsod-ddgc.pkl", 
        "all-elements-fsod-mebv": "all-elements-fsod-mebv.pkl",
        "aquarium-combined-fsod-gjvb": "aquarium-combined-fsod-gjvb.pkl",
        "defect-detection-yjplx-fxobh-fsod-amdi": "defect-detection-yjplx-fxobh-fsod-amdi.pkl",
        "dentalai-i4clz-fsod-fsuo": "dentalai-i4clz-fsod-fsuo.pkl",
        "flir-camera-objects-fsod-tdqp": "flir-camera-objects-fsod-tdqp.pkl",
        "gwhd2021-fsod-atsv": "gwhd2021-fsod-atsv.pkl",
        "lacrosse-object-detection-fsod-uxkt": "lacrosse-object-detection-fsod-uxkt.pkl",
        "new-defects-in-wood-uewd1-fsod-tffp": "new-defects-in-wood-uewd1-fsod-tffp.pkl",
        "orionproducts-vtl2z-fsod-puhv": "orionproducts-vtl2z-fsod-puhv.pkl",
        "paper-parts-fsod-rmrg": "paper-parts-fsod-rmrg.pkl",
        "recode-waste-czvmg-fsod-yxsw": "recode-waste-czvmg-fsod-yxsw.pkl",
        "soda-bottles-fsod-haga": "soda-bottles-fsod-haga.pkl",
        "the-dreidel-project-anzyr-fsod-zejm": "the-dreidel-project-anzyr-fsod-zejm.pkl",
        "trail-camera-fsod-egos": "trail-camera-fsod-egos.pkl",
        "water-meter-jbktv-7vz5k-fsod-ftoz": "water-meter-jbktv-7vz5k-fsod-ftoz.pkl",
        "wb-prova-stqnm-fsod-rbvg": "wb-prova-stqnm-fsod-rbvg.pkl",
        "wildfire-smoke-fsod-myxt": "wildfire-smoke-fsod-myxt.pkl",
        "x-ray-id-zfisb-fsod-dyjv": "x-ray-id-zfisb-fsod-dyjv.pkl"
    }


    os.makedirs(pkl_output_dir, exist_ok=True)

    processed_files = 0
    not_found_in_map = 0

    for json_filename in os.listdir(coco_json_dir):
        if not json_filename.endswith(".json"):
            continue
    
        if "zshot" in json_filename:
            continue

        dataset_name = json_filename.replace(".json", "").replace("preds_", "")

        json_file_path = os.path.join(coco_json_dir, json_filename)

        
        pickle_filename = None
        for prefix, full_pkl_name in pickle_filenames_map.items():
            if prefix.startswith(dataset_name): # Using startswith to mimic original's flexibility if needed
                pickle_filename = full_pkl_name
                
                break
        
        if not pickle_filename:
            continue


        pickle_path = os.path.join(pkl_output_dir, pickle_filename)

        print(f"Processing {json_file_path} -> {pickle_path}")
        ground_truth_file = os.path.join("scratch", "owlvit_data", dataset_name, "test", "_annotations.coco.json")
        with open(ground_truth_file) as f:
            gt_json = json.load(f)

        with open(json_file_path, "r", encoding="utf-8") as f:
            coco_detections = json.load(f) # This is a flat list of detections

        all_detections_by_image = defaultdict(list)
        image_id_to_filename = {z["id"]: z["file_name"] for z in coco_detections["images"]}
        cat_id_to_name = {z["id"]: z["name"] for z in coco_detections["categories"]}

        for det in coco_detections["annotations"]:
            # The structure of 'det' from the COCO JSON output is:
            # {
            #     "image_id": ...,
            #     "category_id": ...,
            #     "bbox": [x, y, w, h],  (This is already a list from JSON)
            #     "score": ...
            # }
            # This structure is already what's needed for the "instances" list.
            image_filename = image_id_to_filename[det["image_id"]]
            category_name = cat_id_to_name[det["category_id"]]
            real_image_id = [z["id"] for z in gt_json["images"] if z["file_name"] == image_filename]
            assert len(real_image_id) == 1
            real_image_id = real_image_id[0]
            real_cat_id = [z["id"] for z in gt_json["categories"] if z["name"] == category_name]
            assert len(real_cat_id) == 1
            real_cat_id = real_cat_id[0]
            all_detections_by_image[real_image_id].append({
                "image_id": real_image_id,
                "category_id": real_cat_id,
                "bbox": list(det["bbox"]), # Ensure it's a list, though JSON load should do this
                "score": float(det["score"])
            })

        pickle_data = []
        for img_id, detections_for_image in all_detections_by_image.items():
            # 'detections_for_image' is already a list of the correctly formatted detection dicts
            image_data = {
                "image_id": img_id,
                "instances": detections_for_image
            }
            pickle_data.append(image_data)
        
        print(dataset_name)
        print(set(z["image_id"] for z in pickle_data))

        try:
            with open(pickle_path, "wb") as f:
                pickle.dump(pickle_data, f)
            print(f"Successfully saved {pickle_path} with {len(pickle_data)} image entries.")
            processed_files += 1
        except Exception as e:
            print(f"Error writing pickle file {pickle_path}: {e}")

    print("\n--- Summary ---")
    print(f"Processed {processed_files} JSON files.")
    if not_found_in_map > 0:
         print(f"{not_found_in_map} datasets used default PKL naming convention.")

def convert_predictions_to_pkl(predictions_dir, output_file):
    """
    Convert predictions from JSON format to the specified PKL format.
    
    Args:
        predictions: List of prediction dictionaries from JSON
        test_info: Test dataset information from JSON
        output_file: Path to save the PKL file
    """
    
    # Group predictions by image_id
    predictions_by_image = defaultdict(list)
    for pred in predictions:
        image_id = pred['image_id']
        
        # Convert bbox format if needed (should already be [x, y, width, height])
        bbox = pred['bbox']
        
        # Create instance dictionary
        instance = {
            'image_id': image_id,
            'category_id': pred['category_id'],
            'bbox': bbox,
            'score': float(pred['score'])
        }
        
        predictions_by_image[image_id].append(instance)
    
    # Create the final result structure
    result = []
    for image_id, instances in predictions_by_image.items():
            
        entry = {
            'image_id': int(image_id),
            'instances': instances
        }
        result.append(entry)
    
    # Save to PKL file
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"Saved {len(result)} images with predictions to {output_file}")
    
    # Print some statistics
    total_instances = sum(len(entry['instances']) for entry in result)
    print(f"Total instances: {total_instances}")
    print(f"Average instances per image: {total_instances / len(result) if result else 0:.2f}")
    
    return result

if __name__ == "__main__":
    # download_projects()
    # benchmark_owlvit()
    # report_stats()
    calculate_map_from_saved_predictions()
    # generate_pkl_from_coco_json("scratch/owlvit_results_new", "scratch/owlvit_pickles")