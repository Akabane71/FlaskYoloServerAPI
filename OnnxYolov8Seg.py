import onnxruntime as ort
import cv2
import numpy as np

def main(image_path):
    # 加载ONNX模型
    ort_session = ort.InferenceSession("./yolov8n-seg.onnx")

    # 预处理输入图像
    input_image = preprocess_image(image_path)

    # 模型输入名称
    input_name = ort_session.get_inputs()[0].name

    # 推理
    outputs = ort_session.run(None, {input_name: input_image})

    # 后处理模型输出
    detections, masks = postprocess_output(outputs)

    # 加载原始图像用于绘制
    original_image = cv2.imread(image_path)

    # 绘制检测框和分割结果
    class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ]

    output_image = draw_detections(original_image, detections, masks, class_names)

    # 展示处理后的图像
    cv2.imshow("YOLOv8 Segmentation Results", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections, masks


def preprocess_image(image_path, input_size=(640, 640)):
    # 读取并预处理图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_size)
    image = image / 255.0  # 归一化到0-1
    image = np.transpose(image, (2, 0, 1))  # 转换为CHW格式
    image = np.expand_dims(image, axis=0).astype(np.float32)  # 增加batch维度
    return image


def postprocess_output(output, conf_threshold=0.5, iou_threshold=0.4):
    # 后处理模型输出
    detections = output[0]
    masks = output[1]

    filtered_detections = []
    for detection in detections:
        confidence = detection[4] if not isinstance(detection[4], np.ndarray) else detection[4][0]
        if confidence >= conf_threshold:
            filtered_detections.append(detection)

    return filtered_detections, masks


def draw_detections(image, detections, masks, class_names, conf_threshold=0.5):
    # 在图像上绘制检测框和红色分割结果
    for i, detection in enumerate(detections):
        x, y, w, h = map(int, detection[:4])
        conf = detection[4]
        class_id = int(detection[5])

        if conf < conf_threshold:
            continue

        # 绘制检测框
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{class_names[class_id]}: {conf:.2f}"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 将掩码二值化，并调整大小
        mask = masks[i]
        print(mask)
        mask_resized = cv2.resize(mask, (w, h))  # 将掩码调整为检测框大小
        mask_binary = (mask_resized > 0.5).astype("uint8")  # 阈值化，生成0/1的掩码

        # 创建红色掩码
        color_mask = np.zeros_like(image[y:y + h, x:x + w])
        color_mask[mask_binary == 1] = [0, 0, 255]  # 使用红色标出掩码

        # 以较高透明度叠加到原始图像上
        image[y:y + h, x:x + w] = cv2.addWeighted(image[y:y + h, x:x + w], 0.5, color_mask, 0.5, 0)

    return image


# 运行示例
main("./2.jpg")
