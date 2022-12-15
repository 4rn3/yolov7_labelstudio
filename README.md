# Setting Up
1. Create a virtual environment for label studio and install the required packages
    >`python -m venv label-studio`

    >`.\label-studio\Scripts\activate`

    >`pip install -r requirements.txt`

2. Clone this repo
   >`git clone https://git.ti.howest.be/TI/2022-2023/s5/project-iv/projects/group01-v2/code.git`
   
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

    >`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117` <br>
    (Note: currently pytorch on windows only supports python versions 3.7-3.9)

    >`label-studio-ml init backend --from .\model_backend.py --force`

    If everything went well you should be able to run the `label-studio-ml start .\backend` command.

6. Installing modules
   > if there are errors about missing modules when running the label-studio-ml backend install them via pip <br>
   > The pip install commands can be found here https://pypi.org/project/ 

7. Enabling ML-assisted Labeling
   >Go to your project, settings > general > Task sampling > select Sequential sampling > click save
   >Go to the Machine Learning tab > ML-Assisted Labeling > enable all 3 buttons > click save
   >Go back to the main project view > click on the order type > Prediction score (make sure the arrow is pointing down)

8. When everything is up and running the .env_example files needs to be renamed to .env . <br>
   >The first field LABEL_STUDIO_HOST is the addres of the label studio instance. <br>
   >The second field LABEL_STUDIO_API_KEY is your label-studio api key this can be found in label studio under account settings (click the cricle with your initials) > Account & Settings > Access Token

9. Bug in YOLOv7 loss.py
   > At the moment (15/12/2022) there seems to be a bug in the loss.py file of the yolov7 implementation. It results in the error: "RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)" on line 759. The fixed that worked for me was to change line 742 to `matching_matrix = torch.zeros_like(cost, device="cpu")`
