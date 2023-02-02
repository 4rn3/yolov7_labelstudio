import os, sys, random, shutil
import glob
import numpy as np
import torch 
from dotenv import load_dotenv
from PIL import Image

import label_studio_sdk

from label_studio_ml import model
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys, is_skipped
from label_studio.core.utils.io import  get_data_dir

model.LABEL_STUDIO_ML_BACKEND_V2_DEFAULT = True

IMG_DATA = './data/images/'
LABEL_DATA = './data/labels/'
INIT_WEIGHTS = './config/checkpoints/starting_weights.pt'#save location for finetuned weights
TRAINED_WEIGHTS = './config/checkpoints/trained_weights.pt'#save location for weights after training
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
REPO = "./yolov7"
IMAGE_SIZE = (640,480)

load_dotenv()
LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST")
LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")

class BloodcellModel(LabelStudioMLBase):
    def __init__(self,  device=DEVICE, img_size=IMAGE_SIZE, repo=REPO, train_output=None, **kwargs):
        super(BloodcellModel, self).__init__(**kwargs)
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')

        self.repo = repo
        self.device = device
        self.image_dir = upload_dir
        self.img_size = img_size
        self.label_map = {}

        if os.path.isfile(TRAINED_WEIGHTS):
            self.weights = TRAINED_WEIGHTS
        else:
            self.weights = INIT_WEIGHTS
        
        print(f"The model initialised with weights: {self.weights}")


        self.model = torch.hub.load(self.repo, 'custom', self.weights, source='local', trust_repo=True)

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

    def move_files(self, files, label_img_data, val_percent=0.3):
        #move files to train or val directories
        print(files)
        print("moving files")
        val_percent = int(len(files)*val_percent)

        #Use last img as val if there are less than 5 imgs
        if len(files) < 5:
            val_file = files[-1]
            base_path = os.path.basename(val_file)
            dest = os.path.join(label_img_data,"val/",base_path)
            shutil.copyfile(val_file, dest)

        for ix, file in enumerate(files):
            train_val = "val/"
            if len(files) - ix > val_percent:
                train_val = "train/"

            base_path = os.path.basename(file)
            dest = os.path.join(label_img_data,train_val,base_path)
            shutil.move(file, dest)

    def reset_train_dir(self, dir_path):
        #remove cache file and reset train/val dir
        if os.path.isfile(os.path.join(dir_path,"train.cache")):
            os.remove(os.path.join(LABEL_DATA, "train.cache"))
            os.remove(os.path.join(LABEL_DATA, "val.cache"))

        for dir in os.listdir(dir_path):
            shutil.rmtree(os.path.join(dir_path, dir))
            os.makedirs(os.path.join(dir_path, dir))
    
    def download_tasks(self, project):
        """
        Download all labeled tasks from project using the Label Studio SDK.
        Read more about SDK here https://labelstud.io/sdk/
        :param project: project ID
        :return:
        """
        ls = label_studio_sdk.Client(LABEL_STUDIO_HOST, LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project)
        tasks = project.get_labeled_tasks()
        return tasks

    def extract_data_from_tasks(self, tasks):
        img_labels = []
        for task in tasks:
                
            if is_skipped(task):
                continue
                        
            image_url = self._get_image_url(task)
            image_path = self.get_local_path(image_url)
            image_name = image_path.split("\\")[-1]
            Image.open(image_path).save(IMG_DATA+image_name)

            img_labels.append(task['annotations'][0]['result'])

            for annotation in task['annotations']:
                for bbox in annotation['result']:
                    bb_width = (bbox['value']['width']) / 100
                    bb_height = (bbox['value']['height']) / 100
                    x = (bbox['value']['x'] / 100 ) + (bb_width/2)
                    y = (bbox['value']['y'] / 100 ) + (bb_height/2)
                    label = bbox['value']['rectanglelabels']
                    label_idx = self.label2idx(label[0])
                        
                    with open(LABEL_DATA+image_name[:-4]+'.txt', 'a') as f:
                        f.write(f"{label_idx} {x} {y} {bb_width} {bb_height}\n")

    def fit(self, tasks, workdir=None, batch_size=8, num_epochs=10, **kwargs):
        print("Start training")
        for dir_path in [IMG_DATA, LABEL_DATA]:
            self.reset_train_dir(dir_path)

        if "data" in kwargs:
            data = kwargs['data']
            project = data['project']['id']
            tasks = self.download_tasks(project)
            self.extract_data_from_tasks(tasks)
        else:
            self.extract_data_from_tasks(tasks)

        img_files = glob.glob(os.path.join(IMG_DATA, "*.jpg"))
        label_files = glob.glob(os.path.join(LABEL_DATA, "*.txt"))

        self.move_files(img_files, IMG_DATA)
        self.move_files(label_files, LABEL_DATA)

        print("start training")
        os.system(f"python ./yolov7/train.py --workers 8 --device {self.device} --batch-size {batch_size} --data ./config/data.yaml --img {self.img_size[0]} {self.img_size[1]} --cfg ./config/model_config.yaml \
                --weights {self.weights} --name bloodcell_trained --hyp ./config/hyp.scratch.p5.yaml --epochs {num_epochs} --exist-ok")

        shutil.move(f"./runs/train/bloodcell_trained/weights/best.pt", TRAINED_WEIGHTS)#move trained weights to checkpoint folder
        print("done training")

        self.weights = TRAINED_WEIGHTS #updating to trained weights
        print(f"The new weights are: {self.weights}")
            
        return {
            'model_path': TRAINED_WEIGHTS,
        }

    def predict(self, tasks, **kwargs):
        print("start predictions")
        print(f"the model uses: {self.weights} to predict")
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
