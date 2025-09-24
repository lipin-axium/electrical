from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parent
    runs = sorted((project_root / "runs").glob("symbol-detector-final*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    model_path = runs[0] if runs else project_root / "runs" / "symbol-detector-final" / "weights" / "best.pt"
    source = project_root / "test3.jpg"
    out_project = project_root / "runs" / "infer"
    out_name = "test3"

    model = YOLO(str(model_path))
    # Ensure class names are correct on plots regardless of what is stored in weights
    # model.names = {0: "Symbol A", 1: "Symbol B"}
    results = model.predict(
        source=str(source),
        project=str(out_project),
        name=out_name,
        save=True,
        imgsz=640,
        verbose=False,
        conf=0.1,
    )
    print(results[0].save_dir)


if __name__ == "__main__":
    main()


