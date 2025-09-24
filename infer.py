from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parent
    runs = sorted((project_root / "runs").glob("symbol-detector-poc*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    model_path = runs[0] if runs else project_root / "runs" / "symbol-detector-poc" / "weights" / "best.pt"
    
    # Test on all demo images (original training images)
    demo_dir = project_root / "demo_images"
    source = demo_dir  # Run inference on entire demo_images directory
    
    out_project = project_root / "runs" / "infer"
    out_name = "demo_images_original"

    print(f"Loading model from: {model_path}")
    print(f"Running inference on: {source}")
    
    model = YOLO(str(model_path))
    # Ensure class names are correct on plots regardless of what is stored in weights
    # model.names is read-only, but we can print the actual names
    print(f"Model class names: {model.names}")
    
    results = model.predict(
        source=str(source),
        project=str(out_project),
        name=out_name,
        save=True,
        imgsz=1280,  # Use same resolution as training
        verbose=True,  # Show detailed results
        conf=0.1,  # Low confidence threshold to catch all symbols
        save_txt=True,  # Save label files
        save_conf=True,  # Save confidence scores
    )
    
    print(f"\n🎯 Inference Results:")
    print(f"Processed {len(results)} images")
    for i, result in enumerate(results):
        img_name = Path(result.path).name
        detections = len(result.boxes) if result.boxes is not None else 0
        print(f"  {img_name}: {detections} symbols detected")
        
        # Print confidence scores for each detection
        if result.boxes is not None and len(result.boxes) > 0:
            for j, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                symbol_name = model.names[cls]
                print(f"    {j+1}. {symbol_name}: {conf:.3f}")
    
    print(f"\nResults saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()


