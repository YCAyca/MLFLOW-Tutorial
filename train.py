import torch
from torch.utils.data import DataLoader
from utils.data import Fruits_Dataset, split_dataset
from utils.augmentation import create_transforms
from utils.trainer import Trainer
from sklearn.metrics import confusion_matrix
from mlflow.models.signature import infer_signature
from utils.helpers import log_dict_as_params, log_system_metrics
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlShutdown
import os 
import mlflow
import os
import yaml
from mlflow.models import infer_signature

""" Prepare training environment """

# Load the YAML configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


dataset_path = config["dataset"]["path"]
dataset_version = config["dataset"]["version"]
dataset_name = config["dataset"]["name"]

input_size = config["input"]["input_size"] 
batch_size = config["input"]["batch_size"] 
seed = config["general"]["seed"]


""" ---- Set your experiment to MLflow server ----"""

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

if mlflow.active_run():
    mlflow.end_run()

mlflow.set_tracking_uri(uri="http://127.0.0.1:9090")

mlflow.set_experiment(config["logging"]["experiment_name"])

# Fix the randomness to generate the different experiments under the same conditions
torch.manual_seed(seed)
generator=torch.Generator().manual_seed(seed)

if config["general"]["device"] == "cuda:0":
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU

# For cuDNN (Optional: Slower but more deterministic)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Create transforms for train and validation using the config
data_transforms = {
        "train": create_transforms(config["data_transforms"]["train"], input_size),
        "val": create_transforms(config["data_transforms"]["val"], input_size),
    }

# Split dataset 
train_list, val_list, test_list, class_idx = split_dataset(data_path = dataset_path)

# Write classnames into outputs folder for inference
class_names = class_idx.keys()

file = open(os.path.join("classes.txt"), "w+")
names = class_idx.keys()
for name in names:
    file.write(name)
    file.write("\n")

# Initialize dataset and dataloaders

train_dataset = Fruits_Dataset(train_list, class_idx, data_transforms['train'])
val_dataset = Fruits_Dataset(val_list, class_idx, data_transforms['val'])
test_dataset = Fruits_Dataset(test_list, class_idx, data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

image_datasets = {"train": train_dataset, "val":val_dataset, "test": test_dataset}
dataloaders = {"train": train_loader, "val":val_loader, "test": test_loader}
dataset_sizes = {"train": len(train_dataset), "val":len(val_dataset), "test": len(test_dataset)}

# Initialize trainer, train and save the finetuned model
import time 
import threading

# Initialize GPU monitoring (if GPU is available)
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0) 

with mlflow.start_run() as run:
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    experiment_name = mlflow.get_experiment(experiment_id).name

    print(f"Experiment Name: {experiment_name}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Run ID: {run_id}")

    mlflow.set_tag("Dataset Name", dataset_name)
    mlflow.set_tag("Dataset Version", dataset_version)

    mlflow.log_dict(config, "config.yaml") # log the full .yaml dictionary as model artifacts
    log_dict_as_params(config) # Log the configs additionally as model parameters overviw section
    if config["logging"]["log_manual_system_metrics"]:
        logging_thread = threading.Thread(target=log_system_metrics, args=(1, gpu_handle))
        logging_thread.daemon = True
        logging_thread.start()
    start_time = time.time()
    trainer = Trainer(config, class_names, dataloaders, dataset_sizes)
    model_finetuned = trainer.train_model()
    end_time = time.time()

   # mlflow.log_metric("train duration", end_time-start_time) NO NEED, MLFLOW logs it by default in "Duration section..."
    if config["logging"]["log_artifacts"]:
        # Log model
        mlflow.pytorch.log_model(model_finetuned, "models/best_model")
    
    # Create input signature for Mlflow
    sample_input = torch.randn(1, 3, config["input"]["input_size"][0], config["input"]["input_size"][1])
    model_finetuned = model_finetuned.to("cpu")
    model_signature = infer_signature(sample_input.numpy(), model_finetuned(sample_input).detach().numpy())
    mlflow.pytorch.log_model(model_finetuned, "models/best_model", signature=model_signature)
    print("--- Start testing the finetuned model ---")
    trainer.test_model()

nvmlShutdown()
