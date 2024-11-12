import cv2
import torch
from torchvision import transforms
from utils.general import non_max_suppression
from utils.augmentations import letterbox
from models.common import DetectMultiBackend

# 加载YOLOv5模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(weights='./weights/yolov5n.pt', device=device)  # 替换为你的权重文件路径
model.eval()

# 自定义的函数，用于调整检测框坐标到原始图像
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords

# 绘制检测框和标签
def plot_boxes(img, boxes, labels):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        label = labels[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 摄像头设置
cap = cv2.VideoCapture(0)
# 设置摄像头的分辨率
width, height = 1920, 1080  # 例如将分辨率设置为1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 确认是否成功设置分辨率
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"摄像头实际分辨率: {actual_width}x{actual_height}")

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取摄像头图像")
        break

    # 显示原始画面
    # print(frame.shape)
    cv2.imshow('Original Camera Feed', frame)

    # 预处理图像
    img = letterbox(frame, new_shape=(640, 640))[0]
    img = img[:, :, ::-1].copy().transpose(2, 0, 1)
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0).to(device)

    # YOLOv5推理
    with torch.no_grad():
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)

    # 处理检测结果并绘制到原始帧上
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f"{model.names[int(cls)]} {conf:.2f}"
                plot_boxes(frame, [xyxy], labels=[label])

    # 显示带有检测框的画面
    cv2.imshow('YOLOv5 Camera Detection', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
