import os
import pickle
import numpy as np
import glob
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Custom dataset class for loading pickle files
class EEGDataset(Dataset):
    def __init__(self, data_dir, n_classes=None, channel_types=None):
        self.file_paths = glob.glob(os.path.join(data_dir, "*.pkl"))
        self.n_classes = n_classes
        self.channel_types = channel_types
        
        if n_classes is not None:
            # Pre-filter files to include only those with valid classes
            valid_files = []
            for file_path in self.file_paths:
                try:
                    with open(file_path, 'rb') as f:
                        sample = pickle.load(f)
                    
                    # Check if the label is within valid range
                    label = sample['y']
                    if isinstance(label, np.ndarray):
                        label = label.item()
                    
                    if 0 <= label < n_classes:
                        valid_files.append(file_path)
                    else:
                        print(f"Skipping file with invalid label {label}: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    
            self.file_paths = valid_files
            print(f"Filtered dataset: {len(self.file_paths)}/{len(glob.glob(os.path.join(data_dir, '*.pkl')))} valid files")
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'rb') as f:
            sample = pickle.load(f)
        
        # Get signal data and channel names
        X = sample['X']
        ch_names = sample.get('ch_names', None)
    
        if self.channel_types is not None and ch_names is not None:
            # Select channels by type (EEG or EOG)
            selected_indices = []
            for i, ch_name in enumerate(ch_names):
                for ch_type in self.channel_types:
                    if ch_type.upper() in ch_name.upper():
                        selected_indices.append(i)
                        break
            
            if selected_indices:
                X = X[selected_indices]
            else:
                print(f"Warning: No {self.channel_types} channels found in {self.file_paths[idx]} - using all channels")
        
        # Convert to torch tensor
        X = torch.tensor(X, dtype=torch.float32)

        # print(X.shape)
        
        # Handle different label formats
        y = sample['y']
        if isinstance(y, np.ndarray):
            y = y.item()
        
        # Final validation
        if self.n_classes and not (0 <= y < self.n_classes):
            raise ValueError(f"Invalid label {y} in {self.file_paths[idx]}, expected 0-{self.n_classes-1}")
            
        return X, torch.tensor(y, dtype=torch.long)

# Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, n_channels, seq_len, n_classes):
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Second convolutional block
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Third convolutional block
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate sizes after each max pooling
        seq_len_after_conv = seq_len // 8  # After 3 max pooling with stride 2
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(64 * seq_len_after_conv, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        )
        
    def forward(self, x):
        # x shape: [batch, channels, sequence_length]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        return x

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.squeeze() if labels.dim() > 1 else labels
        
        # Print input shape, max, min   
        print(inputs.shape, inputs.max(),inputs.min())
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze() if labels.dim() > 1 else labels
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro'
    )
    
    return {
        'loss': running_loss / len(test_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(true_labels, predictions)
    }

def main():
    parser = argparse.ArgumentParser(description='Train a simple CNN on EEG datasets')
    parser.add_argument('--dataset', type=str, 
                        choices=['hmc', 'workload', 'sleepedf', 'seedvig', 'eog', 'isruc'],
                        required=True, help='Dataset to use')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the prepared dataset directory')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Add channel selection arguments
    parser.add_argument('--channels', type=int, nargs='+',
                        help='Indices of channels to use (e.g., 0 1 2)')
    parser.add_argument('--channel_types', type=str, nargs='+', choices=['EEG', 'EOG'],
                        help='Types of channels to use (e.g., EEG EOG)')
    
    args = parser.parse_args()
    
    # Validate channel selection arguments
    if args.channels is not None and args.channel_types is not None:
        parser.error("Please specify either --channels or --channel_types, not both")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    torch.seed()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset specific configurations
    dataset_configs = {
        'hmc': {
            'n_classes': 5,  # Sleep stages: W, N1, N2, N3, REM
            'class_names': ['W', 'N1', 'N2', 'N3', 'REM'] 
        },
        'workload': {
            'n_classes': 2,  # Binary classification (high/low workload)
            'class_names': ['Low Workload', 'High Workload']
        },
        'sleepedf': {
            'n_classes': 7,  # Sleep stages: W, N1, N2, N3, REM, Movement, Undefined
            'class_names': ['W', 'N1', 'N2', 'N3', 'REM']
        },
        'seedvig': {
            'n_classes': 3,  # Vigilance states: Awake, Tired, Drowsy
            'class_names': ['Awake', 'Tired', 'Drowsy']
        },
        'eog': {
            'n_classes': 2,  # Binary classification (saccade vs blink)
            'class_names': ['Blink', 'Saccade']
        },
        "isruc": {
            'n_classes': 5,  # Sleep stages: W, N1, N2, N3, REM
            'class_names': ['W', 'N1', 'N2', 'N3', 'REM']
        }
    }
    
    config = dataset_configs[args.dataset]
    

    if args.channel_types is not None:
        print(f"Using channel types: {args.channel_types}")
    else:
        print("Using all available channels")
    
    # Load datasets
    train_dir = os.path.join(args.data_dir, 'train')
    eval_dir = os.path.join(args.data_dir, 'eval')
    test_dir = os.path.join(args.data_dir, 'test')
    
    print("Loading datasets and filtering invalid labels...")
    train_dataset = EEGDataset(train_dir, config['n_classes'], args.channel_types)
    eval_dataset = EEGDataset(eval_dir, config['n_classes'], args.channel_types)
    test_dataset = EEGDataset(test_dir, config['n_classes'], args.channel_types)
    
    if len(train_dataset) == 0:
        raise ValueError(f"No valid training data found in {train_dir}")
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Eval={len(eval_dataset)}, Test={len(test_dataset)}")
    
    # Create data loaders with error handling
    def collate_fn(batch):
        # Filter out problematic samples
        batch = [(x, y) for x, y in batch if y is not None]
        if not batch:
            raise ValueError("No valid samples in batch")
        return torch.utils.data.dataloader.default_collate(batch)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, 
                             collate_fn=collate_fn, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)
    
    # Get sample shape to initialize model properly
    sample_X, _ = train_dataset[0]
    n_channels, seq_len = sample_X.shape
    print(f"Input shape after channel selection: channels={n_channels}, sequence_length={seq_len}")
    
    # Create model with proper input dimensions
    model = SimpleCNN(n_channels=n_channels, seq_len=seq_len, n_classes=config['n_classes']).to(device)
    print(model)
    
    # Loss function and optimizer
    # For imbalanced datasets like EOG (more saccades than blinks)
    if args.dataset.startswith('eog'):
        # Calculate class weights to handle potential imbalance
        y_train = [int(train_dataset[i][1]) for i in range(len(train_dataset))]
        class_counts = np.bincount(y_train)
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        class_weights = class_weights / class_weights.sum()
        print(f"Using class weights: {class_weights}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("Starting training...")
    
    # Create a results directory with dataset and channel info for saving results
    run_name = f"{args.dataset}"
    if args.channels:
        run_name += f"_ch{'_'.join(map(str, args.channels))}"
    elif args.channel_types:
        run_name += f"_{'_'.join(args.channel_types)}"
    
    train_losses = []
    train_accs = []
    eval_metrics = []
    
    try:
        for epoch in range(args.epochs):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Evaluate on validation set
            metrics = evaluate(model, eval_loader, criterion, device)
            eval_metrics.append(metrics)
            
            print(f"Epoch {epoch+1}/{args.epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Eval Loss: {metrics['loss']:.4f}, Eval Acc: {metrics['accuracy']*100:.2f}%, "
                f"F1: {metrics['f1']:.4f}")
        
        # Final evaluation on test set
        final_metrics = evaluate(model, test_loader, criterion, device)
        print("\nTest Results:")
        print(f"Accuracy: {final_metrics['accuracy']*100:.2f}%")
        print(f"Precision: {final_metrics['precision']:.4f}")
        print(f"Recall: {final_metrics['recall']:.4f}")
        print(f"F1 Score: {final_metrics['f1']:.4f}")
        
        # Print confusion matrix with class names
        cm = final_metrics['confusion_matrix']
        print("\nConfusion Matrix:")
        print(cm)
        
        # Also display a more detailed confusion matrix with class names if available
        if 'class_names' in config:
            class_names = config['class_names']
            print("\nConfusion Matrix with Class Names:")
            cm_df = pd.DataFrame(
                cm, 
                index=class_names, 
                columns=class_names
            )
            print(cm_df)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training curves
        epochs_range = range(1, args.epochs + 1)
        ax1.plot(epochs_range, train_losses, 'b', label='Training loss')
        ax1.plot(epochs_range, [m['loss'] for m in eval_metrics], 'r', label='Validation loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        ax2.plot(epochs_range, train_accs, 'b', label='Training accuracy')
        ax2.plot(epochs_range, [m['accuracy']*100 for m in eval_metrics], 'r', label='Validation accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save results with channel info in the filename
        plt.savefig(f'{run_name}_training_curves.png')
        torch.save(model.state_dict(), f'{run_name}_model.pt')
        
        # Save metrics to file
        with open(f"{run_name}_metrics.txt", "w") as f:
            f.write(f"Dataset: {args.dataset}\n")
            if args.channels:
                f.write(f"Channels: {args.channels}\n")
            elif args.channel_types:
                f.write(f"Channel types: {args.channel_types}\n")
            else:
                f.write("All channels used\n")
            f.write(f"Test Accuracy: {final_metrics['accuracy']*100:.2f}%\n")
            f.write(f"Test Precision: {final_metrics['precision']:.4f}\n")
            f.write(f"Test Recall: {final_metrics['recall']:.4f}\n")
            f.write(f"Test F1 Score: {final_metrics['f1']:.4f}\n")
        
        print(f"Results saved with prefix '{run_name}'")
    
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
        if "CUDA" in str(e):
            print("\nThis could be caused by labels outside the expected range.")
            print("Try checking your data or adding CUDA_LAUNCH_BLOCKING=1 to your environment variables for more details.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

if __name__ == "__main__":
    main()
