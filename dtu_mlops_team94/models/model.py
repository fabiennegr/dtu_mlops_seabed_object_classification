from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
import wandb
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights, ResNet18_Weights
import pytorch_lightning as pl

class SeabedClassifier(pl.LightningModule):
    def __init__(self, num_classes, lr, optimizer_name):
        """Seabed classifier model to be used as a base model for transfer learning.

        Args:
            num_classes: number of classes in the dataset
            lr: learning rate
        """
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr  # learning rate
        self.optimizer_name = optimizer_name
        # self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(self.model.fc.in_features, 256), torch.nn.ReLU(), torch.nn.Linear(256, self.num_classes)
        )

        # Freeze all the parameters in the network
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last fully-connected layer (our output layer)
        for param in self.model.fc.parameters():
            param.requires_grad = True

        # log hyperparameters to wandb
        self.save_hyperparameters()

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x: input tensor expected to be of shape [N,C,H,W]

        Returns:
            Output tensor with shape [N,num_classes]
        """
        return self.model(x)
    
    def accuracy(self, y_pred, y):
        """Accuracy metric.

        Args:
            y_pred: predictions
            y: ground truth

        Returns:
            Accuracy
        """
        return torch.sum(torch.argmax(y_pred, dim=1) == y).item() / len(y)    

    def training_step(self, batch, batch_idx):
        """Training step for the model.

        Args:
            batch: batch of data
            batch_idx: index of the batch

        Returns:
            Loss tensor
        """
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.cross_entropy(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        # Log training loss
        self.log('train_loss', loss)
    
        # Log metrics
        self.log('train_acc', accuracy)

        # Print training loss and accuracy
        print(f"Training Loss: {loss.item()}, Training Accuracy: {accuracy}\n")

        #wandb.log({"training_loss": loss})

        return loss
    
    def validation_step(self, batch, batch_idx):
        '''used for logging metrics'''
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.cross_entropy(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        # Log validation loss (will be automatically averaged over an epoch)
        self.log('valid_loss', loss)

        # Log accuracy
        self.log('valid_acc', accuracy)

        # Print validation loss and accuracy
        print(f"Validation Loss: {loss.item()}, Validation Accuracy: {accuracy}\n")

    def test_step(self, batch, batch_idx):
        """Testing step for the model.

        Args:
            batch: batch of data
            batch_idx: index of the batch

        Returns:
            Loss tensor
        """
        x, y = batch
        y_pred = self(x)
        loss = torch.nn.functional.cross_entropy(y_pred, y)
        accuracy = self.accuracy(y_pred, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', accuracy)

        # Print test loss and accuracy
        print(f"Test Loss: {loss.item()}, Test Accuracy: {accuracy}\n")

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:

        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        return optimizer
