from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import  models
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix

class Trainer():
    def __init__(self, config, classnames, dataloaders, dataset_sizes):
        self.train_loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []
        self.epochs = []
        self.class_names = classnames
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes

        self.model_type = config["model"]["type"]
        self.device = config["general"]["device"]
        self.num_epochs = config["training"]["epochs"]
        self.optimizer = config["training"]["optimizer"]
        self.criterion = config["training"]["criterion"]
        self.learning_rate = config["training"]["learning_rate"]
        self.step_size = config["training"]["scheduler"]["step_size"]
        self.gamma = config["training"]["scheduler"]["gamma"]
        self.early_stopped = False

        # mlflow.log_param("model type",self.model_type)
        # mlflow.log_param("class names",self.class_names)
        # mlflow.log_param("optimizer", self.optimizer)
        # mlflow.log_param("learning rate", self.learning_rate)
        # mlflow.log_param("max epochs",self.num_epochs)
        # mlflow.log_param("device",self.device)
                

        if self.model_type == "InceptionV3":   # Load Pre-trained InceptionV3
            model_ft = models.inception_v3(pretrained=True)
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, len(self.class_names))

            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, len(self.class_names))

            model_ft = model_ft.to(self.device)
            self.model = model_ft
        else:  # Other models can be added here if we want to experiment them
            raise AssertionError("Model type not implemented yet")
        
        # Define Loss Function and Optimizer
        if self.optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        else:
            raise AssertionError("Optimizer type not implemented yet")
        
        if self.criterion == "CrossEntropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise AssertionError("Criterion type not implemented yet")
        
         
        # Learning Rate Scheduler
        self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    # Training the Model
    def train_model(self):
        best_model_wts = self.model.state_dict()
        best_acc = 0.0

        early_stopping_patience = 5
        early_stopping_counter = 0
        best_val_loss = float('inf')
            
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)
            self.epochs.append(epoch)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for batch in tqdm(self.dataloaders[phase]):
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward
                    # Track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        if phase == 'train':
                            outputs, aux_outputs = outputs
                            loss1 = self.criterion(outputs, labels)
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2  # Weighted sum of main and auxiliary loss
                        else:
                            loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                if phase == "train":
                    self.scheduler.step()
                    self.train_loss.append(epoch_loss)
                    self.train_accuracy.append(epoch_acc.cpu())
                    mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                    mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)
                else:
                    self.val_loss.append(epoch_loss)
                    self.val_accuracy.append(epoch_acc.cpu())
                    mlflow.log_metric("val_loss", epoch_loss, step=epoch)
                    mlflow.log_metric("val_accuracy", epoch_acc, step=epoch)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.model.state_dict()

                if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_acc = epoch_acc
                        best_model_wts = self.model.state_dict()
                        early_stopping_counter = 0  # Reset the counter if we get a better loss
                else:
                    early_stopping_counter += 1

            # Early stopping check
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch} due to no improvement in validation loss for {early_stopping_patience} consecutive epochs.')
                self.early_stopped = True  # Hypothetical attribute; framework-dependent
                break

        
            # Plot and save
            plt.figure(figsize=(5, 5), num=1)
            plt.clf()
            plt.plot(self.epochs, self.train_loss, label='Train')
            plt.plot(self.epochs, self.val_loss, label='Test')
            plt.legend()
            plt.grid()
            plt.title('Cross entropy loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            mlflow.log_figure(plt.gcf(), "loss.png")

            plt.figure(figsize=(5, 5), num=2)
            plt.clf()
            plt.plot(self.epochs, self.train_accuracy, label='Train')
            plt.plot(self.epochs, self.val_accuracy, label='Test')
            plt.legend()
            plt.grid()
            plt.title('Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            mlflow.log_figure(plt.gcf(), "accuracy.png")

        print(f'Best val Acc: {best_acc:4f}')

        if self.early_stopped:
            mlflow.log_param("early_stopping", True)
            mlflow.log_param("stopped_epoch", epoch)
        else:
            mlflow.log_param("early_stopping", False)


        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model
            
    def test_model(self):
        self.model.eval()
        self.model.to(self.device)

        epoch_total_correct = 0
        epoch_total_samples = 0
        test_accuracy = 0
        
        y_pred = []
        y_true = []

        with torch.no_grad():
            for batch in tqdm(self.dataloaders["test"]):
                imgs, labels = batch[0].to(self.device), batch[1].to(self.device)

                outputs = self.model(imgs.float())
                _, predicted = torch.max(outputs.data, 1)
            
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

                epoch_total_samples += labels.size(0)
                epoch_total_correct += (predicted == labels).sum().item()


            test_accuracy = epoch_total_correct / epoch_total_samples

    
        mlflow.log_metric("test_accuracy", test_accuracy)
        " Create confisuion matrix and save it as artifacts "
        # Handle confusion matrix with all classes
        cf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(self.class_names))))
        panda_matrix = pd.DataFrame(
        cf_matrix,
        index=self.class_names,  # Rows: True Labels
        columns=self.class_names  # Columns: Predicted Labels
        )

        plt.figure(figsize = (12,7))
        sn.heatmap(panda_matrix, annot=True)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")

