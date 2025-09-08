from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: Optional[DataLoader],

        optimizer,
        criterion,
        scheduler = None,
        early_stopping = None,

        step_callback = None,
        epoch_callback = None,
        
        device: str = "cuda",
        verbose: bool = True,
    ):
        # Assigning values to instance
        self.model = model

        # Setting device
        if device == "cuda" and torch.cuda.is_available():
            self.model.to(device)
            self.device = device
        else:
            self.model.to("cpu")
            self.device = "cpu"
        if verbose:
            print(f"Device of trainer is {self.device}")

        # Dataset
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        # Regularizer
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        
        # Call back during training
        if step_callback is not None:
            assert isinstance(step_callback, function)
            self.step_callback = step_callback
        else:
            self.step_callback = lambda epoch, step, loss: print(f"--- Epoch {epoch} Step {step} ; Train Loss {loss} ---")
        
        # Call back after one epoch
        if epoch_callback is not None:
            assert isinstance(epoch_callback, function)
            self.epoch_callback = epoch_callback
        else:
            self.epoch_callback = lambda epoch, train_loss, val_loss, val_acc: print(f"EPOCH {epoch} Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_acc}")

        # History
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
    
    def train(
        self,
        epochs = 30,
        step_callback_interval: int = 50
    ):
        # Set state
        self.model.train()

        # Loop over epochs
        for epoch in range(epochs):
            epoch_loss = 0.0
            # Loop over sample
            for i, sample in enumerate(self.train_loader):
                step_loss = self._fit_train_sample(sample)
                epoch_loss += step_loss
                
                # Call back after an interval
                if (i + 1) % step_callback_interval == 0:
                    self.step_callback(epoch = epoch, step = i + 1, loss = step_loss)
            
            train_loss = epoch_loss / len(self.train_loader)
            self.train_losses.append(train_loss)

            # Evaluation and other stuff
            val_loss, val_acc = self._evaluation()
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            if self.early_stopping is not None:
                self.early_stopping(train_loss, val_loss)

            # Call back
            callback_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc
            }
            self.epoch_callback(**callback_dict)
    
    def _fit_train_sample(self, sample):
        data, label = sample
        
        # Get input to device
        data = data.to(self.device)
        label = label.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        prediction = self.model(data)
        loss_item = self.criterion(prediction, label)

        # Backward pass and update weights
        loss_item.backward()
        self.optimizer.step()
        return loss_item.item()
    
    def _evaluation(self):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss, total_samples = 0.0, 0

        with torch.no_grad():
            for data, label in self.val_loader:
                data, label = data.to(self.device), label.to(self.device)
                outputs = self.model(data)

                loss = self.criterion(outputs, label)
                total_loss += loss.item() * data.size(0)   # scale by batch size
                total_samples += data.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        val_loss = total_loss / total_samples
        val_acc = accuracy_score(all_labels, all_preds)

        return val_loss, val_acc
    
    def validate(self, class_names=["0", "1"], plot_roc=True, use_val = True):
        # Choose loader
        if use_val:
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader

        val_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_preds = []
        all_probs = []

        self.model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)  # shape [B, C]

                if labels.dim() == 2:
                    labels_idx = labels.argmax(dim=1)
                else:
                    labels_idx = labels.view(-1)

                loss = self.criterion(outputs, labels_idx)
                val_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(probs, 1)

                correct += (preds == labels_idx).sum().item()
                total += inputs.size(0)

                if plot_roc:
                    all_labels.extend(labels_idx.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())  # shape: [batch, num_classes]

        avg_loss = val_loss / total
        accuracy = correct / total * 100

        # Plot ROC curve if desired
        if plot_roc:
            # Convert to numpy
            all_labels_np = np.array(all_labels)
            all_preds_np = np.array(all_preds)
            all_probs_np = np.array(all_probs)
        
            # Classification report and confusion matrix
            print("\nClassification Report:")
            print(classification_report(all_labels_np, all_preds_np, target_names=class_names))
        
            print("Confusion Matrix:")
            print(confusion_matrix(all_labels_np, all_preds_np))
        
            # ROC AUC (macro-average over all classes)
            try:
                auc = roc_auc_score(
                    np.eye(len(np.unique(all_labels_np)))[all_labels_np],
                    all_probs_np,
                    multi_class='ovr',
                    average='macro'
                )
                print(f"AUC-ROC Score (macro): {auc:.4f}")
            except ValueError as e:
                print(f"ROC AUC couldn't be calculated: {e}")
                auc = None

            if auc is not None:
                fpr = dict()
                tpr = dict()
                for i in range(all_probs_np.shape[1]):
                    fpr[i], tpr[i], _ = roc_curve((all_labels_np == i).astype(int), all_probs_np[:, i])
                    plt.plot(fpr[i], tpr[i], label=f"Class {class_names[i] if class_names else i}")
        
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                plt.grid()
                plt.show()
        
        return avg_loss, accuracy