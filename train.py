import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(model=r'D:\YOLO-UXO\pythonProject1\ultralytics\ultralytics\cfg\models\11\MFAE-YOLO11.yaml')
    model.load('yolo11s.pt')
    model.train(data=r'D:\YOLO-UXO\pythonProject1\ultralytics\images\data.yaml',
                imgsz=640,
                epochs=100,
                batch=16,
                workers=8,
                device='1',
                optimizer='SGD',
                close_mosaic=15,
                resume=False,
                project='runs/train',
                name='UXO-MFAE',
                single_cls=False,
                cache='True',
                amp=False,
                )

