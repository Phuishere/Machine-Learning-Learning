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

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0015)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 2) # giảm sau mỗi 2 lần không cải thiện (thường fluctuate 1 lần)
criterion = nn.CrossEntropyLoss()

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: Optional[DataLoader],

        optimizer,
        criterion,
        early_stopping,
        scheduler,

        step_callback,
        epoch_callback,
        
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
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        
        # Call back during training
        self.step_callback = step_callback
        self.epoch_callback = epoch_callback

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
                epoch_loss += step_loss.item()
                
                # Call back after an interval
                if (i + 1) % step_callback_interval == 0:
                    self.step_callback(epoch = i + 1, loss = step_loss)
            
            train_loss = epoch_loss / len(self.train_loader)
            val_loss, val_acc = self.train_losses.append(train_loss)

            self._evaluation(self)
            
            # Call back
            callback_dict = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc
            }
            self.epoch_callback(callback_dict)
    
    def _fit_train_sample(self, sample):
        data, label = sample
        
        # Get input to device
        data = data.to(self.device)
        label = label.to(self.device)

        self.optimizer.zero_grad()

        # Đưa dữ liệu theo chiều xuôi (forward pass)
        prediction = self.model(data)
        loss_item = self.criterion(prediction, label)

        # Đạo hàm ngược để cập nhật tham số mô hình (backward pass và update weights)
        loss_item.backward()
        optimizer.step()
        return loss_item.item()
    
    def _evaluation(self):
        # Set state
        self.model.eval()
        
        # Looping over validation set
        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():  # No gradient
            for data, label in self.val_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                outputs = self.model(data)

                # Loss calculation
                loss = self.criterion(outputs, label)  # assume self.criterion is defined
                total_loss += loss.item()
                num_batches += 1

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        # Get loss
        val_loss = total_loss / len(self.val_loader)
        self.val_losses.append(val_loss)
        
        # Get accuracy
        val_acc = accuracy_score(all_labels, all_preds)
        self.val_accs.append(val_acc)

        return val_loss, val_acc
    
    def validate(self, class_names=[0, 1], plot_roc=True):
        self.model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
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