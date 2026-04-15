from ultralytics import YOLO

# 1. 加载训练好的模型权重（替换为你的最佳权重路径）
best_model_path = r"D:\2025-03 ANTIUAV Project\ultralytics\runs\train\yolov11n-antiuav\weights\best.pt"  # 或使用last.pt
model = YOLO(best_model_path)

# 2. 执行预测
results = model.predict(
    source=r"D:\2025-03 ANTIUAV Project\ultralytics\ANTI-UAV Data\images\test",  # 测试集图片路径
    conf=0.25,          # 置信度阈值
    iou=0.7,            # IOU阈值
    save=True,          # 保存带检测框的可视化图片
    save_txt=True,      # 保存检测结果为YOLO格式的txt标签
    save_conf=True,     # 在txt文件中保存置信度
    save_crop=False,    # 是否保存裁剪的检测目标
    project='runs/detect',  # 结果保存目录
    name='yolov11n_test(1)',  # 实验名称
    exist_ok=True,      # 允许覆盖现有目录
    device='1'          # 使用指定GPU
)

# 可选：打印结果统计信息
for result in results:
    print(f"检测到 {len(result.boxes)} 个目标")