import os, sys
import numpy as np
import torch 
from PIL import Image

from label_studio_ml import model
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys, is_skipped, get_choice
from label_studio.core.utils.io import json_load, get_data_dir

sys.path.insert(1, './yolov7/')
from models.experimental import attempt_load

model.LABEL_STUDIO_ML_BACKEND_V2_DEFAULT = True

YOLO_REPO = '../yolov7'
WEIGHTS = './fine_tuned_results/tiny-100epoch-bs8/yolov7-tiny-1OOepoch.pt'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = (640,480)

class BloodcellModel(LabelStudioMLBase):
    def __init__(self, weights=WEIGHTS,  device=DEVICE, img_size=IMAGE_SIZE, yolo_repo=YOLO_REPO, train_output=None, **kwargs):
        super(BloodcellModel, self).__init__(**kwargs)
        self.device = device
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = upload_dir
        self.img_size = img_size
        self.label_map = {}
        self.yolo_repo = yolo_repo
        self.model = attempt_load(weights, map_location=device)
        
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image'
        )
        
        schema = list(self.parsed_label_config.values())[0]
        self.labels_in_config = set(self.labels_in_config)
        
        self.label_attrs = schema.get('labels_attrs')
        if self.label_attrs:
            for label_name, label_attrs in self.label_attrs.items():
                for predicted_value in label_attrs.get('predicted_values', '').split(','):
                    self.label_map[predicted_value] = label_name
    
    def _get_image_url(self,task):
        image_url = task['data'][self.value]
        return image_url

    def fit(self, tasks, workdir=None, batch_size=16, num_epochs=10, **kwargs):
        image_urls, image_labels = [], []
        for task in tasks:
            
            if is_skipped(task):
                continue
            
            image_path = self.get_local_path(task['data'][self.value])

            for annotation in task['annotations']:
                for bbox in annotation['result']:
                    x = bbox['value']['x']
                    y = bbox['value']['y']
                    width = bbox['value']['width']
                    height = bbox['value']['height']
                    label = bbox['value']['rectanglelabels']

            #TODO:
            #pixel norm: x/self.img[x] , width/self.img[x] <- same for y,height
            #map class to 0 -> 2
            #save imgs in data/img/train
            #save labels as class x_center y_center width height in data/label/train with same name as corresponding img
            #test the other way of loading/ training model
            #change weight loading path to runs/name/weights/fine_tuned.pt

        return {'model_path': "./", 'labels': ""}
    
    def predict(self, tasks, **kwargs):

        results = []
        all_scores= []
        print("LABELS IN CONFIG:",self.labels_in_config)
        for task in tasks:
           
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url, project_dir=self.image_dir)
            img = Image.open(image_path)
            img_width, img_height = get_image_size(image_path)
            
            preds = self.model(img, size=img_width)
            preds_df = preds.pandas().xyxy[0]
            
            for x_min,y_min,x_max,y_max,confidence,class_,name_ in zip(preds_df['xmin'],preds_df['ymin'],
                                                                        preds_df['xmax'],preds_df['ymax'],
                                                                        preds_df['confidence'],preds_df['class'],
                                                                        preds_df['name']):
                #add label name from label_map
                output_label = self.label_map.get(name_, name_)
                print("--"*20)
                print(f"Output Label {output_label}")
                if output_label not in self.labels_in_config:
                    print(output_label + ' label not found in project config.')
                    continue
                
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    "original_width": img_width,
                    "original_height": img_height,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [name_],
                        'x': x_min / img_width * 100,
                        'y': y_min / img_height * 100,
                        'width': (x_max - x_min) / img_width * 100,
                        'height': (y_max - y_min) / img_height * 100
                    },
                    'score': confidence
                })
                all_scores.append(confidence)
                print(results)

        avg_score = sum(all_scores) / max(len(all_scores), 1)
        
        return [{
            'result': results,
            'score': avg_score
        }]
