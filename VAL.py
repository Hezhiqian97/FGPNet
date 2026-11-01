import warnings

warnings.filterwarnings("ignore")

from FGP import YOLO

if __name__ == "__main__":
    model = YOLO("runs/FGPNet_exp/exp/weights/best.pt")
    model.val(
        data="data/InfraRedSmallTarget.yaml",
        split="val",
        imgsz=640,
        batch=16,
        iou=0.6,
        rect=False,
        save_json=False,
        project="runs/val",
        name="exp",
    )
