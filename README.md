# Setting Up
1. Create a virtual environment for label studio 
    >`python -m venv label-studio`

    >`.\label-studio\Scripts\activate or on linux: source ./env/bin/activate`

2. Clone this repo & install the required packages
   >`https://github.com/4rn3/yolov7_labelstudio.git`<br>
   > `cd code` <br>
   >`pip install -r requirements.txt` <br>

   
3. Setting up label studio
   >`git clone https://github.com/heartexlabs/label-studio.git`

   >`pip install label-studio`

   Label studio is properly set up if using the `label-studio` command opens label studio in a new browser tab.

4. Git clone the yolov7 repo
   > `git clone https://github.com/WongKinYiu/yolov7.git`

5. Setting up label studio ml backend
    >`git clone https://github.com/heartexlabs/label-studio-ml-backend`

    >`cd label-studio-ml-backend`

    >`pip install -U -e .`

    >`cd ..`

    >`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117` <br>
    (Note: currently pytorch on windows only supports python versions 3.7-3.9)

    >`label-studio-ml init backend --from .\model_backend.py --force`

    If everything went well you should be able to run the `label-studio-ml start .\backend` command, if there are errors look at point 6 first.

6. Installing missing modules
   > if there are errors about missing modules when running the label-studio-ml backend install them via pip <br>
   > The pip install commands can be found here https://pypi.org/project/ 

7. Setting up the .env file <br>
   > rename the .example_env file to .env <br>
   >The first field LABEL_STUDIO_HOST is the addres of the label studio instance. <br>
   >The second field LABEL_STUDIO_API_KEY is your label-studio api key this can be found in label studio under account settings (click the cricle with your initials) > Account & Settings > Access Token

8. Connecting the backend
   >Go to your project, settings > general > Task sampling > select Sequential sampling > click save <br>
   >Go to the labeling interface to setup the labels, add Platelets in the "add labels names" box and click "Add". Repeat this for RBC and WBC <br>
   >Start the backend with `label-studio-ml start .\backend` (it should start without errors)<br>
   >Head over to settings > Machine Learning and click "Add Model" <br>
   >Give the backend a title and add its address i.e http://localhost:9090 <br>
   >(optional) set a description <br>
   >select "Use for interactive preannotation" <br>
   >In the "ML-Assisted Labeling" section enable all 3 buttons <br>
   >Click save <br>
   >Go back to the main project view > click on the order type > Prediction score (make sure the arrow is pointing down) <br>

   Now the images should be preannotated, the preannotated images are then ordered based on score to more easily improve the bad annotations. The model is updated with each submition or update of an annotation.

9. Bug in YOLOv7 loss.py
   > At the moment (15/12/2022) there seems to be a bug in the loss.py file of the yolov7 implementation. It results in the error: "RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)" on line 759. The fixed that worked for me was to change line 742 to `matching_matrix = torch.zeros_like(cost, device="cpu")`
