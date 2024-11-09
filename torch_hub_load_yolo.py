"""
    2024-11-09 测试成功！
    可以加载本地的权重文件
"""

import torch
from PIL import Image

# 设置本地YOLOv5项目路径
local_yolov5_path = "E:\PythonProject\YOLOv5\yolov5-master"  # 替换为你YOLOv5项目的实际路径
model = torch.hub.load(local_yolov5_path, 'custom', path='./weights/yolov5n.pt', source='local')

# 'custom' 代表是自定义权重， path 设置为你本地权重文件的路径
# 'source' 设置为 'local' 表示从本地加载

# 读取和加载图片
img_path = "./2.jpg"  # 替换为你实际的图片路径
img = Image.open(img_path).convert("RGB")

# 使用模型预测
results = model(img)
print(results.pandas().xyxy[0].to_json(orient="records")) # 返回json数据
# 查看预测结果
results.print()  # 打印预测信息
results.show()   # 显示带有预测框的图片
