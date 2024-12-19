import argparse
import mlflow.pytorch
import torch
from PIL import Image
from torchvision import transforms
import json
import os
import time
import matplotlib.pyplot as plt

def main(args):
    # Load the model from MLflow
    mlflow.set_tracking_uri(uri="http://127.0.0.1:9090")
    model_uri = f"runs:/{args.run_id}/models/best_model"
    print(f"Loading model from: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri)

    # Define the input transformation
    transform = transforms.Compose([
        transforms.Resize((229, 229)),
        transforms.ToTensor(),
    ])

    # Perform inference
    model.eval()

    images = os.listdir(args.input_path)

    file = open(args.class_names_file, "r")
    lines =file.readlines()
    classnames = [name.strip() for name in lines]


    with mlflow.start_run(run_id=args.run_id):
        with torch.no_grad():
            av_inf_time = 0
            av_fps = 0
            for i, image_name in enumerate(images):
                # Load and preprocess the input image
                start = time.time()
                image = Image.open(os.path.join(args.input_path, image_name))
                input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
                # Inference the input image
                prediction = model(input_tensor)
                end = time.time()
                inference_time = end-start
                fps = 1 / inference_time

                av_inf_time += inference_time
                av_fps += fps

                mlflow.log_metric("inference_time_in_sec", inference_time, step=i)
                mlflow.log_metric("fps",fps, step=i)
        
                predicted_label = torch.argmax(prediction, dim=1).item()



                print(f"Predicted label: {classnames[predicted_label]}")

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(image)
                ax.axis("off")  # Turn off axes
                ax.set_title(f"Prediction: {classnames[predicted_label]}", fontsize=16, color="blue")


                if args.output_dir:
                    # Save results
                    fig.savefig(os.path.join(args.output_dir, image_name))

                # Log inference results to MLflow
                mlflow.log_figure(fig, "inference_torch/"+image_name)
                plt.clf()
                plt.close(fig)


            av_inf_time /= len(images)
            av_fps /= len(images)
            mlflow.log_metric("average inference time per image", av_inf_time)
            mlflow.log_metric("average fps",av_fps)
            mlflow.log_param("inferenced",True)
        
    return

if __name__ == "__main__":
     # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference with a model from MLflow")
    parser.add_argument("--run_id",default="a25b0f4891734a6ba942090ea2f445a0", type=str, help="MLflow Run ID of the model")
    parser.add_argument("--input_path", default="data/inference_dataset", type=str, help="Path to the input images")
    parser.add_argument("--class_names_file",  default="classes.txt", help=".txt file containing class names, same order with training")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save inference results")
    args = parser.parse_args()
   
    main(args)
    
    
    
    
