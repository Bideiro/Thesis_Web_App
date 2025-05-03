from ultralytics import YOLO

def evaluate_yolo_model(model_path, data_yaml_path, device='cuda'):
    # Load model
    model = YOLO(model_path)
    
    # Validate
    metrics = model.val(data=data_yaml_path, device=device, verbose=True)
    
    # Extract proper metrics using methods
    # acc = metrics.map50()      # mAP@0.5
    # prec = metrics.mp()        # mean precision
    # rec = metrics.mr()         # mean recall
    # f1 = metrics.f1() if hasattr(metrics, 'f1') else 2 * (prec * rec) / (prec + rec + 1e-16)  # fallback if needed

    # Speed metrics
    inference_time_ms = metrics.speed['inference']  # ms per image
    yolo_fps = 1000 / inference_time_ms if inference_time_ms else 0
    avg_prediction_time = inference_time_ms / 1000  # seconds

    avg_memory_usage = 0  # Not measured
    total_memory_usage = 0

    # # Print
    # print("\n=== Model Performance Metrics ===")
    # print(f"ResNet Metrics:")
    # print(f"  - Accuracy (mAP50):         {acc*100:.2f}%")
    # print(f"  - Precision:                {prec*100:.2f}%")
    # print(f"  - Recall:                   {rec*100:.2f}%")
    # print(f"  - F1 Score:                  {f1*100:.2f}%\n")
    print("Performance Metrics (YOLO only):")
    print(f"  - YOLO FPS:                  {yolo_fps:.2f}")
    print(f"  - Average Prediction Time (Including ResNet): {avg_prediction_time:.4f} seconds\n")
    print("Memory Usage Metrics:")
    print(f"  - Average Memory Usage:      {avg_memory_usage:.2f} MB")
    print(f"  - Total Memory Usage:        {total_memory_usage:.2f} MB")
    print("=================================\n")

if __name__ == "__main__":
    evaluate_yolo_model(r'runs/detect/YOLOv5s(Synthetic_Cleaned)_e10_30e_2025-04-27/weights/best.pt',
                        r'd:\Documents\ZZ_Datasets\Synthetic_Cleaned_FINAL(4-21-25)\data.yaml')
    # evaluate_yolo_model(r'runs/detect/YOLOv8s(Synthetic_Cleaned)_e10_30e_2025-04-22/weights/best.pt',
    #                   r'd:\Documents\ZZ_Datasets\Synthetic_Cleaned_FINAL(4-21-25)\data.yaml')