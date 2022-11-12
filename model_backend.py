import os, sys, random, shutil
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

IMG_DATA = './data/images/'
LABEL_DATA = './data/labels/'
WEIGHTS = './config/checkpoints/starting_weights.pt'#save location for finetuned weights
MODEL_PATH = './config/checkpoints/trained_weights.pt'#save location for weights after training
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = (640,480)

class BloodcellModel(LabelStudioMLBase):
    def __init__(self, weights=WEIGHTS,  device=DEVICE, img_size=IMAGE_SIZE, train_output=None, **kwargs):
        super(BloodcellModel, self).__init__(**kwargs)
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        
        if os.path.isfile(MODEL_PATH):
            self.weights = MODEL_PATH
        else:
            self.weights = weights
        
        self.device = device
        self.image_dir = upload_dir
        self.img_size = img_size
        self.label_map = {}
        self.model = attempt_load(self.weights, map_location=self.device)
        
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

    def label2idx(self,label):
        #convert label to according index in data.yaml
        if label == 'Platelets':
            return 0
        if label == "RBC":
            return 1
        return 2

    def move_files(self,files, percent, train_val="train/", img_label=IMG_DATA):
        #move new img, label to train/val dir
        if train_val=="train/":
            start = 0
            end = len(files)-percent 
        else: 
            start= len(files)-percent
            end = len(files)

        for file in files[start:end]:
            shutil.move(os.path.join(img_label,file), os.path.join(img_label+train_val,file) )

    def init_train_val(self):
        #clean up left over training files
        if len(os.listdir(IMG_DATA +"/train/")) >0:
            shutil.rmtree(IMG_DATA+"/train/")
            shutil.rmtree(IMG_DATA+"/val/")
            shutil.rmtree(LABEL_DATA+"/train/")
            shutil.rmtree(LABEL_DATA+"/val/")

            os.makedirs(IMG_DATA+"/train/")
            os.makedirs(IMG_DATA+"/val/")
            os.makedirs(LABEL_DATA+"/train/")
            os.makedirs(LABEL_DATA+"/val/")

    def fit(self, tasks, workdir=None, batch_size=16, num_epochs=10, **kwargs):
        print("start training")
        self.init_train_val()
        for task in tasks:
            
            if is_skipped(task):
                continue
            
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url)
            image_name = image_path.split("\\")[-1]

            Image.open(image_path).save(IMG_DATA+image_name)
            #TODO check why img size changed to 320 240
            for annotation in task['annotations']:
                for bbox in annotation['result']:
                    x_center = bbox['value']['x'] / 100
                    y_center = bbox['value']['y'] / 100
                    width = bbox['value']['width'] / 100
                    height = bbox['value']['height'] / 100
                    label = bbox['value']['rectanglelabels']
                    label_idx = self.label2idx(label[0])
                    #TODO check if coords are correct now
                    with open(LABEL_DATA+image_name[:-5]+'.txt', 'a') as f:
                        f.write(f"{label_idx} {x_center} {y_center} {width} {height}\n")
        
        img_files = os.listdir(IMG_DATA)[:-2]
        val_percent = int(len(img_files)*0.3)
        
        self.move_files(img_files,val_percent)
        self.move_files(img_files,val_percent,train_val="val/")

        label_files = os.listdir(LABEL_DATA)[:-2]
        self.move_files(label_files,val_percent,img_label=LABEL_DATA)
        self.move_files(label_files,val_percent,train_val="val/",img_label=LABEL_DATA)

        os.system(f"python ./yolov7/train.py --workers 8 --device {self.device} --batch-size {batch_size} --data ./config/data.yaml --img 320 240 --cfg ./config/model_config.yaml \
            --weights {self.weights} --name bloodcell_trained --hyp ./config/hyp.scratch.p5.yaml --exist-ok")

        #shutil.move(f"./runs/train/bloodcell_trained/best.pt", MODEL_PATH)#move trained weights to checkpoint folder

        return {'model_path': MODEL_PATH}
    
    def predict(self, tasks, **kwargs):
        print("start predictions")
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
                print(f'height:{(y_max - y_min) / img_height * 100}')
                print(f'width:{(x_max - x_min) / img_width * 100}')
                print(results)

        avg_score = sum(all_scores) / max(len(all_scores), 1)
        
        return [{
            'result': results,
            'score': avg_score
        }]
