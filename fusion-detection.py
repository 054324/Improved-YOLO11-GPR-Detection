import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from abc import ABC, abstractmethod
import threading
import queue


class VideoFusionDetector:
    def __init__(self, config):
        self.config = config
        self.frame_queue = queue.Queue(maxsize=30)  # 帧处理队列
        self.running = True
        self.initialize()

    def initialize(self):
        """初始化所有组件"""
        print("初始化视频检测系统...")

        # 创建视频捕获对象
        self.cap_ir = self.create_video_capture(self.config['ir_video_path'])
        self.cap_vis = self.create_video_capture(self.config['vis_video_path'])

        # 获取视频属性
        self.fps_ir = self.cap_ir.get(cv2.CAP_PROP_FPS)
        self.fps_vis = self.cap_vis.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap_ir.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap_ir.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = min(int(self.cap_ir.get(cv2.CAP_PROP_FRAME_COUNT)),
                                int(self.cap_vis.get(cv2.CAP_PROP_FRAME_COUNT)))

        print(f"红外视频: {self.width}x{self.height}, {self.fps_ir:.2f} FPS, {self.total_frames} 帧")
        print(f"可见光视频: {self.width}x{self.height}, {self.fps_vis:.2f} FPS, {self.total_frames} 帧")

        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*self.config['video_codec'])
        self.fused_writer = self.create_video_writer(
            self.config['fused_output_path'], fourcc, min(self.fps_ir, self.fps_vis),
            (self.width, self.height))

        self.detected_writer = self.create_video_writer(
            self.config['detected_output_path'], fourcc, min(self.fps_ir, self.fps_vis),
            (self.width, self.height))

        # 新增：四格布局视频写入器
        quad_width = int(2 * self.width * self.config['display_scale'])
        quad_height = int(2 * self.height * self.config['display_scale'])
        self.quad_writer = self.create_video_writer(
            self.config['quad_output_path'], fourcc, min(self.fps_ir, self.fps_vis),
            (quad_width, quad_height))

        # 初始化融合器和检测器
        self.fuser = self.config['fusion_module'](self.config['pyramid_levels'])
        self.detector = self.config['detection_module'](self.config['yolo_model_path'])

        # 性能计数器
        self.frame_count = 0
        self.start_time = time.time()
        self.detect_times = []
        self.fuse_times = []

        # 创建处理线程
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True

    def create_video_capture(self, path):
        """创建视频捕获对象"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"视频文件不存在: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {path}")
        return cap

    def create_video_writer(self, path, fourcc, fps, frame_size):
        """创建视频写入对象"""
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        return cv2.VideoWriter(path, fourcc, fps, frame_size)

    def capture_frames(self):
        """捕获帧的线程函数"""
        print("开始捕获视频帧...")
        while self.running:
            ret_ir, ir_frame = self.cap_ir.read()
            ret_vis, vis_frame = self.cap_vis.read()

            if not ret_ir or not ret_vis:
                break

            # 调整可见光帧尺寸
            if vis_frame.shape[:2] != (self.height, self.width):
                vis_frame = cv2.resize(vis_frame, (self.width, self.height))

            # 转换红外帧为灰度（如果必要）
            if len(ir_frame.shape) == 3:
                ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)

            # 将帧放入队列
            try:
                self.frame_queue.put((ir_frame, vis_frame), timeout=1.0)
            except queue.Full:
                print("帧队列已满，跳过一帧")
                continue

        self.running = False
        print("视频捕获完成")

    def process_frames(self):
        """处理帧的线程函数"""
        print("开始处理视频帧...")
        while self.running or not self.frame_queue.empty():
            try:
                ir_frame, vis_frame = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                if not self.running:
                    break
                continue

            # 图像融合
            fuse_start = time.time()
            fused_frame = self.fuser.fuse(ir_frame, vis_frame)
            fuse_time = time.time() - fuse_start
            self.fuse_times.append(fuse_time)

            # 保存融合结果
            self.fused_writer.write(fused_frame)

            # 目标检测
            detect_start = time.time()
            detected_frame = self.detector.detect(fused_frame)
            detect_time = time.time() - detect_start
            self.detect_times.append(detect_time)

            # 保存检测结果
            self.detected_writer.write(detected_frame)

            # 计算性能指标
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            detect_fps = 1.0 / detect_time if detect_time > 0 else 0

            # 创建四格布局显示帧
            display_frame = self.config['display_module'].create_display(
                ir_frame, vis_frame, fused_frame, detected_frame,
                fps, detect_fps, self.config['display_scale']
            )

            # 新增：保存四格布局视频
            self.quad_writer.write(display_frame)

            # 显示结果
            if self.config['display_process']:
                cv2.imshow(self.config['window_title'], display_frame)

                # 检查退出按键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

            # 显示进度
            if self.frame_count % 50 == 0:
                progress = self.frame_count / self.total_frames * 100
                avg_fps = self.frame_count / elapsed_time
                print(f"处理进度: {progress:.1f}% | 帧: {self.frame_count}/{self.total_frames} | FPS: {avg_fps:.1f}")

            self.frame_queue.task_done()

        print("视频处理完成")

    def run(self):
        """主处理循环"""
        # 启动捕获线程
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.daemon = True
        capture_thread.start()

        # 启动处理线程
        self.processing_thread.start()

        # 等待线程完成
        capture_thread.join()
        self.processing_thread.join()

        # 释放资源
        self.release()

        # 打印性能报告
        self.print_performance_report()

    def release(self):
        """释放所有资源"""
        print("释放资源...")
        self.running = False

        # 等待队列清空
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get(timeout=0.5)
                self.frame_queue.task_done()
            except queue.Empty:
                break

        self.cap_ir.release()
        self.cap_vis.release()
        self.fused_writer.release()
        self.detected_writer.release()
        self.quad_writer.release()  # 新增：释放四格布局写入器
        cv2.destroyAllWindows()

    def print_performance_report(self):
        """打印性能报告"""
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0

        avg_fuse_time = np.mean(self.fuse_times) * 1000 if self.fuse_times else 0
        avg_detect_time = np.mean(self.detect_times) * 1000 if self.detect_times else 0
        max_detect_time = np.max(self.detect_times) * 1000 if self.detect_times else 0

        print("\n" + "=" * 60)
        print("视频检测处理完成 - 性能报告")
        print("=" * 60)
        print(f"总帧数: {self.frame_count}/{self.total_frames}")
        print(f"总时间: {total_time:.2f}秒")
        print(f"平均FPS: {avg_fps:.2f}")
        print(f"平均融合时间: {avg_fuse_time:.2f}毫秒")
        print(f"平均检测时间: {avg_detect_time:.2f}毫秒")
        print(f"最大检测时间: {max_detect_time:.2f}毫秒")
        print(f"融合视频保存至: {os.path.abspath(self.config['fused_output_path'])}")
        print(f"检测视频保存至: {os.path.abspath(self.config['detected_output_path'])}")
        print(f"四格布局视频保存至: {os.path.abspath(self.config['quad_output_path'])}")  # 新增
        print("=" * 60)


class FusionAlgorithm(ABC):
    """图像融合算法抽象基类"""

    @abstractmethod
    def fuse(self, ir_img, vis_img):
        """融合红外和可见光图像"""
        pass


class LaplacianPyramidFuser(FusionAlgorithm):
    """拉普拉斯金字塔融合算法"""

    def __init__(self, pyramid_levels=4):
        self.pyramid_levels = pyramid_levels

    def fuse(self, ir_img, vis_img):
        """使用拉普拉斯金字塔融合红外和可见光图像帧"""
        # 转换红外图像为三通道（如果需要）
        if len(ir_img.shape) == 2:
            ir_img = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)

        ir_img = ir_img.astype(np.float32)
        vis_img = vis_img.astype(np.float32)

        # 亮度通道融合
        vis_yuv = cv2.cvtColor(vis_img, cv2.COLOR_BGR2YUV)
        vis_y, vis_u, vis_v = cv2.split(vis_yuv)

        ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
        ir_y = ir_gray.astype(np.float32)

        fused_y = self.fuse_single_channel(ir_y, vis_y)

        fused_yuv = cv2.merge([fused_y, vis_u, vis_v])
        fused_img = cv2.cvtColor(fused_yuv, cv2.COLOR_YUV2BGR)

        return np.clip(fused_img, 0, 255).astype(np.uint8)

    def fuse_single_channel(self, img1, img2):
        """融合单通道图像"""
        pyramid_levels = self.pyramid_levels

        # 构建高斯金字塔
        G1 = [img1]
        G2 = [img2]
        for _ in range(pyramid_levels - 1):
            G1.append(cv2.pyrDown(G1[-1]))
            G2.append(cv2.pyrDown(G2[-1]))

        # 构建拉普拉斯金字塔
        L1 = [G1[-1]]
        L2 = [G2[-1]]

        for i in range(len(G1) - 2, -1, -1):
            expanded1 = cv2.pyrUp(G1[i + 1], dstsize=(G1[i].shape[1], G1[i].shape[0]))
            expanded2 = cv2.pyrUp(G2[i + 1], dstsize=(G2[i].shape[1], G2[i].shape[0]))
            L1.append(G1[i] - expanded1)
            L2.append(G2[i] - expanded2)

        L1.reverse()
        L2.reverse()

        # 融合金字塔
        fused_pyramid = [(L1[i] + L2[i]) / 2 for i in range(pyramid_levels)]

        # 重建融合图像
        fused_img = fused_pyramid[-1]
        for i in range(pyramid_levels - 2, -1, -1):
            fused_img = cv2.pyrUp(fused_img, dstsize=(fused_pyramid[i].shape[1], fused_pyramid[i].shape[0]))
            fused_img += fused_pyramid[i]

        return np.clip(fused_img, 0, 255)


class FastAverageFuser(FusionAlgorithm):
    """快速平均融合算法（替代方案）"""

    def fuse(self, ir_img, vis_img):
        """快速融合算法，适合实时处理"""
        # 转换红外图像为三通道（如果需要）
        if len(ir_img.shape) == 2:
            ir_img = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)

        # 简单加权平均
        fused = cv2.addWeighted(ir_img, 0.5, vis_img, 0.5, 0)
        return fused


class DetectionAlgorithm(ABC):
    """目标检测算法抽象基类"""

    @abstractmethod
    def detect(self, image):
        """在图像上执行目标检测"""
        pass


class YOLODetector(DetectionAlgorithm):
    """YOLO目标检测器"""

    def __init__(self, model_path):
        # 初始化参数
        self.model = YOLO(model_path)
        self.device = 'cuda' if self.model.device.type != 'cpu' else 'cpu'
        print(f"加载YOLO模型: {model_path} | 设备: {self.device}")

    def detect(self, image):
        """执行目标检测并返回带标注的图像"""
        # 使用半精度推理加速
        results = self.model(image, verbose=False, half=True, imgsz=640)
        return results[0].plot()


class DisplayModule:
    """显示模块基类"""

    def create_display(self, ir_frame, vis_frame, fused_frame, detected_frame,
                       fps, detect_fps, scale=1.0):
        """创建显示图像"""
        pass


class QuadDisplay(DisplayModule):
    """四格布局显示模块"""

    def create_display(self, ir_frame, vis_frame, fused_frame, detected_frame,
                       fps, detect_fps, scale=0.7):
        """创建四格布局的显示图像"""
        # 按比例缩放所有帧
        ir_display = cv2.resize(ir_frame, None, fx=scale, fy=scale)
        vis_display = cv2.resize(vis_frame, None, fx=scale, fy=scale)
        fused_display = cv2.resize(fused_frame, None, fx=scale, fy=scale)
        detected_display = cv2.resize(detected_frame, None, fx=scale, fy=scale)

        # 确保红外图为三通道
        if len(ir_display.shape) == 2:
            ir_display = cv2.cvtColor(ir_display, cv2.COLOR_GRAY2BGR)

        # 创建两行布局
        top_row = np.hstack((ir_display, vis_display))
        bottom_row = np.hstack((fused_display, detected_display))
        combined = np.vstack((top_row, bottom_row))

        # 添加标签
        font_scale = 0.7 * scale
        thickness = max(1, int(2 * scale))
        height = ir_display.shape[0]
        width = ir_display.shape[1]

        cv2.putText(combined, f"Infrared (FPS: {fps:.1f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        cv2.putText(combined, "Visible", (width + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        cv2.putText(combined, "Fused Result", (10, height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        cv2.putText(combined, f"Detection (FPS: {detect_fps:.1f})",
                    (width + 10, height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

        # 添加分隔线
        line_thickness = max(1, int(2 * scale))
        cv2.line(combined, (width, 0), (width, combined.shape[0]), (0, 255, 0), line_thickness)
        cv2.line(combined, (0, height), (combined.shape[1], height), (0, 255, 0), line_thickness)

        # 添加进度条
        progress_width = int(combined.shape[1] * 0.8)
        progress_height = 10
        progress_x = (combined.shape[1] - progress_width) // 2
        progress_y = combined.shape[0] - 20

        if hasattr(self, 'frame_count') and hasattr(self, 'total_frames'):
            progress = self.frame_count / self.total_frames
            cv2.rectangle(combined, (progress_x, progress_y),
                          (progress_x + progress_width, progress_y + progress_height),
                          (100, 100, 100), -1)
            cv2.rectangle(combined, (progress_x, progress_y),
                          (progress_x + int(progress_width * progress), progress_y + progress_height),
                          (0, 200, 0), -1)

        return combined


class MinimalDisplay(DisplayModule):
    """最小化显示模块（仅显示检测结果）"""

    def create_display(self, ir_frame, vis_frame, fused_frame, detected_frame,
                       fps, detect_fps, scale=1.0):
        """仅显示检测结果"""
        # 缩放检测结果
        display_frame = cv2.resize(detected_frame, None, fx=scale, fy=scale)

        # 添加FPS信息
        font_scale = 0.7 * scale
        thickness = max(1, int(2 * scale))
        cv2.putText(display_frame, f"FPS: {fps:.1f} | Detect FPS: {detect_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)

        return display_frame


if __name__ == "__main__":
    # ===================== 配置参数 =====================
    config = {
        # 视频路径
        'ir_video_path': r"C:\Users\Administrator\Desktop\M2VD-main\M2VD-main\Video\test\infrared\0114_1611.mp4",
        'vis_video_path': r"C:\Users\Administrator\Desktop\M2VD-main\M2VD-main\Video\test\visible\0114_1611.mp4",

        # 输出路径
        'fused_output_path': "outputs/fused_video.avi",
        'detected_output_path': "outputs/detected_video.avi",
        'quad_output_path': "outputs/quad_display_video.avi",  # 新增：四格布局视频输出路径

        # 算法参数
        'pyramid_levels': 4,
        'yolo_model_path': r"D:\2025-03 ANTIUAV Project\ultralytics\runs\train\yolov11n_train_multimodal\weights\best.pt",
        # 替换为你的模型路径

        # 显示设置
        'display_process': True,
        'display_scale': 0.7,
        'window_title': "红外可见光融合视频检测系统",
        'video_codec': 'XVID',

        # 模块选择
        'fusion_module': LaplacianPyramidFuser,  # 可替换为FastAverageFuser
        'detection_module': YOLODetector,
        'display_module': QuadDisplay()  # 可替换为MinimalDisplay()
    }

    # ===================== 运行系统 =====================
    try:
        print("=" * 60)
        print("红外可见光融合视频检测系统")
        print("=" * 60)

        detector = VideoFusionDetector(config)
        detector.run()

        print("处理完成!")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback

        traceback.print_exc()
    finally:
        # 确保程序退出前关闭所有资源
        if 'detector' in locals():
            detector.release()