import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from FGP import YOLO

# 模型配置文件（仓库内自带，无需修改）
model_yaml_path = str(Path(__file__).parent / "FGP" / "cfg" / "models" / "11" / "FGP-seg.yaml")

# 数据集配置文件（按本机实际路径修改）
data_yaml_path = r"F:\dataset\FIDMS_YOLO_721\dataset.yaml"
# data_yaml_path = r"F:\dataset\SAM\NEU_Seg-main\NEU_YOLO\dataset.yaml"
# data_yaml_path = r"F:\dataset\SAM\MSD\MSD_YOLO\MSD\data.yaml"
# data_yaml_path = r"F:\dataset\CSDD_seg\CSDD_YOLO\dataset.yaml"

if __name__ == "__main__":
    model = YOLO(model_yaml_path)
    # model.load("yolo26n-seg.pt")  # 加载预训练权重，如果模型改动很大就不加载预训练权重（自选）
    # 训练模型
    results = model.train(
        data=data_yaml_path,  # 指定数据集的配置文件
        imgsz=640,  # 输入图像的尺寸，所有输入图像将被缩放到 640x640 像素
        epochs=200,  # 训练的轮数
        batch=12,  # 每次训练的批次大小
        workers=10,
        # amp=False,         # 如果出现训练损失为Nan可以关闭amp
        device="0",  # device="0,1,2,3" 训练时使用的 GPU 设备编号，通常设置为 0
        project="runs/FGPNet_exp",  # 保存训练结果文件夹
        name="exp",
    )
