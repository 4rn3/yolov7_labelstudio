import os 
import cv2
import torch 
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys

from label_studio.core.utils.io import json_load, get_data_dir 

YOLO_REPO = "WongKinYiu/yolov7"
WEIGHTS = './fine_tuned_results/tiny-100epoch-bs8/yolov7-tiny-1OOepoch.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    
    def predict(self, tasks, **kwargs):

        results = []
        all_scores= []
        print("LABELS IN CONFIG:",self.labels_in_config)
        for task in tasks:
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url, project_dir=self.image_dir)
            ori_h, ori_w = image.shape[:2]
            image = cv2.resize(image_path, (self.img_size[0], self.img_size[1]))
            image = torch.from_numpy(image).permute(2,0,1).to(self.device)
            image = image.float() / 255.0
            preds = self.model(image)
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
                    "original_width": ori_w,
                    "original_height": ori_h,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [name_],
                        'x': x_min / ori_w * 100,
                        'y': y_min / ori_h * 100,
                        'width': (x_max - x_min) / ori_w * 100,
                        'height': (y_max - y_min) / ori_h * 100
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