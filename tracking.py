from rich.progress import track

from ultralytics import YOLO
model= YOLO(model=r"D:\2025-03 ANTIUAV Project\ultralytics-main\runs\train\yolov11n-antiuav2\weights\best.pt")

results = model.track(source=r"C:\Users\Administrator\Desktop\59d5df14f88deea12cc95b7da926b9cc.mp4",show=True,
                      tracker=r"D:\2025-03 ANTIUAV Project\ultralytics-main\ultralytics\cfg\trackers\bytetrack.yaml")