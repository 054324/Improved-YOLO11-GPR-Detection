# CMDAF-YOLO11
The content is keep updating.
## A Quick Overview 

## Start
Firstly, you should prepare the dataset for training or testing. Please follow these steps to get started:
Follow the next steps for training or testing.
1. Dataset Preparing: Follow the instructions in the Dataset Preparing section to organize your GPR B-scan images.
2. Pretrained Weights: If you wish to test or use our pretrained CMDAF-YOLO11, please download and prepare the .pt files as instructed in the Pretrained Weights section.
3. Training: Once the dataset and environment are ready, you can run the train.py file to start the training process.

## Dataset Preparing <a id="dataset_section"></a>
You can download our preproccessed dataset from the follow link: https://pan.baidu.com/s/1rNJP7vNLOOO6YqISSotAMw?pwd=bqnz Extracted code: bqnz.

Instructions: The dataset is formatted in standard YOLO format. After downloading, please extract the files into the datasets/ directory of the project and ensure the paths match those defined in your Multimodal data.yaml.

## Pretrained Weights <a id="weights_section"></a>
We provide several .pt files pretrained on diverse geological datasets. Please place the downloaded .pt files in the weights/ directory:

yolo11-cmdaf-uxo.pt: Optimized for detecting Unexploded Ordnance (UXO) hyperbolic signatures in complex subsurface clutter.

yolo11-cmdaf-general.pt: A general-purpose model for various GPR underground targets (pipes, cavities, etc.).

## Execution
**Training**
To start training with the CMDAF-enhanced architecture:
python train.py --model cfg/models/11/yolo11-cmdaf.yaml --data data/Multimodal_data.yaml
We have exported our virtual environment configuration in conda, You can use our configuration file, requirements.txt, to quickly deploy the runtime environment.
Since multiple projects share the same virtual environment, the list of packages is quite extensive.
**Inference**
To run detection on a single GPR B-scan image or a directory:
python detection.py --weights weights/yolo11-cmdaf-uxo.pt --source examples/test_gpr.jpg
## Example Cases

## TODO LIST


## Who do I talk to
youyou-Wang, Nanjing University of Aeronautics and Astronautics
*** 15202856246@163.com ***

## Cite

Ultralytics YOLO11: Jocher, G., & Qiu, J. (2024). Ultralytics YOLO11 (Version 11.0.0). [Software]. Available at https://github.com/ultralytics/ultralytics.

MFAE-YOLO (Inspiration): Liu, Y., et al. "MFAE-YOLO: Multi-Feature Attention-Enhanced Network for Remote Sensing Images Object Detection." IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1-14, 2025. doi: 10.1109/TGRS.2025.3583467.
