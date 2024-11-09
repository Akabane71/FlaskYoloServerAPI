import atexit
import threading

import cv2
from flask import Flask, Response, render_template, g, request, flash, redirect, url_for, jsonify
from torchvision import transforms
from utils.general import *
from models.common import *
import counter

app = Flask(__name__, static_url_path='/static')

Light = counter.LightControl(30)  # 最长10s就熄灭


class YOLOv5():
    def __init__(self, weights='weights/yolov5n.pt'):
        self.weights = weights
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.model = DetectMultiBackend(weights=self.weights)
        self.model.to(self.device)

    def draw(self, image, x1, x2, y1, y2, cls, conf):
        # print('x1:', x1)
        # print('y1:', y1)
        # print('x2', x2)
        # print('y2', y2)
        color = (0, 255, 0)  # BGR格式，这里表示绿色
        thickness = 2  # 边界框线条粗细
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # 在图像上绘制边界框
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        text = f'Class: human, Confidence: {conf:.2f}'
        cv2.putText(image, text, (int(x1), int(y1) - 5), font, font_scale, color, font_thickness)
        return image

    def frame_to_tensor(self, frame):

        frame = cv2.resize(frame, (640, 640))
        # 将图像转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将图像转换为 张量 并归一化
        transform = transforms.Compose([transforms.ToTensor()])
        frame_tensor = transform(frame_rgb).unsqueeze(0)  # 添加批量维度

        return frame_tensor

    def get_frame(self, frame):

        # 图片预处理
        frame_tensor = self.frame_to_tensor(frame)

        device = "cuda" if torch.cuda.is_available() else 'cpu'
        frame_tensor = frame_tensor.to(device)

        # 使用模型进行推理
        with torch.no_grad():
            outputs = self.model(frame_tensor)

            # 极大值抑制 ----> 去除框的数量,筛选置信度最高的
            pred = non_max_suppression(outputs, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=1000)

        # 排除为空的情况
        if pred:
            for detection in pred:
                # 每个 detection 是一个 tensor，每行是一个边界框，包括 (x1, y1, x2, y2, conf, cls)
                for box in detection:
                    box_list = box.tolist()
                    # print(box_list)   # 调试
                    if len(box_list) >= 6:
                        x1, y1, x2, y2, conf, cls = box_list
                        # print(f'Bounding box: ({x1}, {y1}) - ({x2}, {y2}), Confidence: {conf}, Class: {cls}')
                    # 筛别人
                    if cls == 0.0:
                        frame = self.draw(frame, x1, x2, y1, y2, cls, conf)
                        Light.turnOn()
                        print("灯的状态",Light.state)

        return frame


# 单例摄像头类，用于共享摄像头资源
class CameraSingleton:
    _instance = None
    _lock = threading.Lock()  # 确保线程安全

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance.camera = cv2.VideoCapture(0)
        return cls._instance

    def get_frame(self):
        ret, frame = self.camera.read()
        return ret, frame

    def release(self):
        if self.camera.isOpened():
            self.camera.release()
            print("摄像头资源已释放")


# 获取摄像头实例
def get_camera():
    return CameraSingleton()


# 定义视频流生成器
def generate_video_stream(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            # 使用multipart格式传输视频流
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break


def generate_frames(camera, yolo):
    num = 0
    while True:
        num += 1
        # 读取视频帧
        ret, frame = camera.get_frame()
        if ret:
            # 在这里可以对视频帧进行处理，例如添加滤镜、人脸识别等
            frame = cv2.resize(frame, (640, 640))  # 解决图像畸变

            # 将处理后的视频帧转换为字节流
            frame = yolo.get_frame(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # 以字节流的形式发送视频帧
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            break


# 使用 atexit 注册释放函数
def release_camera_at_exit():
    camera = get_camera()
    camera.release()


atexit.register(release_camera_at_exit)


@app.route('/submit', methods=['POST'])  # @ 叫做装饰器 简化一个写法
def submit():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        print(username)
        print(password)
        if username and password:
            if username == 'admin' and password == 'admin':
                # 返回包含<video>标签的HTML页面
                return render_template('index.html')
        else:
            return redirect(url_for('login'))


# 登录界面
@app.route('/')
def login():
    return render_template('login.html')


@app.route('/video_feed')
def video_feed():
    camera = get_camera()
    yolo = YOLOv5()
    return Response(generate_frames(camera, yolo), mimetype='multipart/x-mixed-replace; boundary=frame')


# 灯光的状态
@app.route('/lightbulb_status', methods=['GET'])
def lightbulb_status():
    LightStatic = Light.checkTime()
    print(LightStatic)
    return jsonify(LightStatic)


# 处理404错误页面
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=8080)
