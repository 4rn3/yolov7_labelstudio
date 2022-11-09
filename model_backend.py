import os
import numpy as np
import torch 
from PIL import Image

from label_studio_ml import model
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys
from label_studio.core.utils.io import json_load, get_data_dir

model.LABEL_STUDIO_ML_BACKEND_V2_DEFAULT = True

YOLO_REPO = "WongKinYiu/yolov7"
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
        self.model = torch.hub.load(yolo_repo, 'custom', weights, trust_repo=True)
        
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

    def fit(self, tasks, workdir=None, **kwargs):
    # Retrieve the annotation ID from the payload of the webhook event
    # Use the ID to retrieve annotation data using the SDK or the API
    # Do some computations and get your model
        return {'checkpoints': './'}
    ## JSON dictionary with trained model artifacts that you can use later in code with self.train_output
    
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