from ultralytics import YOLO
import onnx
import tensorrt as trt

def pt2onnx():
    # 加载预训练的 YOLOv8 模型，或者指定你自己的模型文件路径
    model = YOLO("./yolov8n-seg.pt")  # 使用官方提供的 YOLOv8n 模型（nano 版本）

    # 导出模型到 ONNX 格式
    # 导出模型到指定路径
    model.export(format="onnx")

def checkOnnx():
    # 加载导出的 ONNX 模型
    onnx_model = onnx.load("./yolov8n-seg.onnx")

    # 检查模型是否正确
    onnx.checker.check_model(onnx_model)
    print("ONNX 模型导出成功并且有效！")


def checkEngine():
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    def load_engine(trt_runtime, engine_path):
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        return trt_runtime.deserialize_cuda_engine(engine_data)

    engine_path = "./yolov8n-seg.engine"  # 你的 TensorRT 引擎文件路径
    runtime = trt.Runtime(TRT_LOGGER)
    engine = load_engine(runtime, engine_path)

    if engine:
        print("TensorRT 已成功加载引擎，可以正常运行。")
    else:
        print("TensorRT 引擎加载失败。")
