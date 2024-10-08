import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy


class PilotCareTransNet(pl.LightningModule):
    def __init__(self, input_dim: int, num_labels: int, learning_rate: float = 1e-4):
        super(PilotCareTransNet, self).__init__()

        # Transformer model architecture for EEG data
        self.input_dim = input_dim
        self.num_labels = num_labels

        # Temporal Convolutional Layer to capture local temporal dependencies
        self.temporal_conv = nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, padding=1)

        # Multi-head Attention mechanism from Transformer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

        # Positional Encoding to maintain sequence information
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, 256))

        # Fully connected layers for classification
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_labels)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Hyperparameters
        self.learning_rate = learning_rate

        # Metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_dim)

        # Temporal Convolutional Layer (apply over the time dimension)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, sequence_length)
        x = self.temporal_conv(x)  # (batch_size, 256, sequence_length)
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, 256)

        # Add positional encoding
        x += self.positional_encoding[:, :x.size(1), :]

        # Multi-head Attention Layer
        attn_output, _ = self.multihead_attn(x, x, x)  # Self-attention
        attn_output = self.dropout(attn_output)

        # Fully connected layers for classification
        x = torch.relu(self.fc1(attn_output[:, 0, :]))  # Using the first token (CLS-like behavior)
        x = self.fc2(self.dropout(x))

        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch['eeg_data'], batch['labels']

        # Forward pass
        logits = self(inputs)
        loss = self.criterion(logits, labels)

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['eeg_data'], batch['labels']

        # Forward pass
        logits = self(inputs)
        loss = self.criterion(logits, labels)

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
