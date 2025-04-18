from ultralytics import YOLO  # 确保ultralytics库已经安装

def train():
    # 加载模型
    # model = YOLO("0417.yaml")  # 加载YOLOv8模型配置
    model = YOLO("0417.yaml")  # 加载YOLOv8模型配置
    # 训练模型
    results = model.train(data="mycoco(1).yaml", epochs=100)  # 指定数据集配置文件和训练轮数

if __name__ == "__main__":
    train()