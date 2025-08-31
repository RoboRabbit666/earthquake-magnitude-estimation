# %% [markdown]
# Cell 1: Initial Setup and Imports

# %%
# -*- coding: utf-8 -*-
"""INSTANCE_50_Experiments_Event_Based_Splits_Runs_1_to_25.ipynb

This notebook runs experiments 1 to 25 with different event-based random splits
of the INSTANCE dataset.
"""

# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
import random
import seaborn as sns
from tqdm import tqdm
from google.colab import drive
import pickle

# Helper function to convert numpy types to Python types for JSON serialization
def numpy_to_python(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [numpy_to_python(i) for i in obj]
    else:
        return obj

# Define the range of split seeds for this notebook
START_SEED = 1
END_SEED = 25

# Define the offset for random seeds to avoid overlap with seismogram-based split seeds
RANDOM_SEED_OFFSET = 50  # This will map split_seed 1→51, 2→52, etc.

# Mount Google Drive if using Colab
drive.mount('/content/drive')

# Configure environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Record start time
start_time = time.time()

# Define paths to data files (UPDATED FOR INSTANCE)
base_dir = "/content/drive/My Drive/2023-2024/UCL MSc in DSML/Term 3/MSc Project/Code/INSTANCE_Event_Based"
all_data_file = os.path.join(base_dir, "all_data.pt")
all_labels_file = os.path.join(base_dir, "all_labels.pt")
split_info_file = os.path.join(base_dir, "event_split_info.pkl")
output_dir = os.path.join(base_dir, "experiment_results")
os.makedirs(output_dir, exist_ok=True)

# Check if files exist
assert os.path.isfile(all_data_file), f"Data file not found at {all_data_file}"
assert os.path.isfile(all_labels_file), f"Labels file not found at {all_labels_file}"
assert os.path.isfile(split_info_file), f"Split info file not found at {split_info_file}"

print("✓ INSTANCE event-based data files found")
print(f"✓ Output directory: {output_dir}")

# %% [markdown]
# Cell 2: Dataset and Model Classes

# %%
#------------------------------------------------------------------------------
# Dataset and Model Classes (FIXED FOR INSTANCE DATA FORMAT)
#------------------------------------------------------------------------------

class EarthquakeDataset(Dataset):
    """Dataset class for earthquake data."""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class EarthquakeModel(nn.Module):
    """MagNet architecture for earthquake magnitude estimation - ADAPTED FOR INSTANCE FORMAT."""
    def __init__(self):
        super(EarthquakeModel, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(4, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(32, 100, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(200, 2)  # Output: [magnitude_prediction, log_variance]

    def forward(self, x):
        # INSTANCE data format: [batch, channels, time_steps] - NO TRANSPOSE NEEDED
        # STEAD data format would be: [batch, time_steps, channels] - would need transpose

        # Check input shape and print for debugging
        # print(f"Input shape: {x.shape}")  # The input format should be [batch, channels, time_steps] which would be (batch_size, 3, 3000)

        # For INSTANCE: x is already [batch, channels, time_steps], so use directly
        # For STEAD: x would be [batch, time_steps, channels], so would need: x = x.transpose(1, 2)

        # Since INSTANCE data is (362234, 3, 3000), each batch will be [batch, 3, 3000]
        # which is exactly what Conv1d expects: [batch, channels, time_steps]

        # First conv block
        x = self.conv1(x)  # Input: [batch, 3, 3000] -> Output: [batch, 64, 3000]
        x = self.dropout(x)
        x = self.maxpool(x)  # Output: [batch, 64, 750]

        # Second conv block
        x = self.conv2(x)  # Input: [batch, 64, 750] -> Output: [batch, 32, 750]
        x = self.dropout(x)
        x = self.maxpool(x)  # Output: [batch, 32, 187]

        # Prepare for LSTM: [batch, time_steps, features]
        x = x.transpose(1, 2)  # [batch, 32, 187] -> [batch, 187, 32]

        # LSTM layer
        x, _ = self.lstm(x)  # Input: [batch, 187, 32] -> Output: [batch, 187, 200]

        # Get the last output of the LSTM
        x = x[:, -1, :]  # [batch, 187, 200] -> [batch, 200]

        # Output layer with magnitude prediction and uncertainty
        x = self.fc(x)  # [batch, 200] -> [batch, 2]

        return x

# %% [markdown]
# Cell 3: Training Components

# %%
#------------------------------------------------------------------------------
# Training Components
#------------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=7, verbose=False, delta=0, run_id=None,
                 split_num=None, model_seed=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.run_id = run_id
        self.split_num = split_num
        self.model_seed = model_seed
        self.best_model_path = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})')
        self.best_model_path = os.path.join(
            output_dir, f'best_model_Run_{self.run_id}_split_{self.split_num}_seed_{self.model_seed}.pth'
        )
        torch.save(model.state_dict(), self.best_model_path)
        self.val_loss_min = val_loss

def custom_loss(y_pred, y_true):
    """
    Custom loss function combining prediction error and uncertainty.

    This implements a negative log-likelihood loss with learned aleatoric uncertainty:
    L = 0.5 * exp(-s) * (y_true - y_hat)^2 + 0.5 * s

    where:
    - y_hat is the predicted magnitude
    - s is the log variance (uncertainty)
    - y_true is the true magnitude

    This loss encourages the model to predict accurate magnitudes while
    also learning to estimate its own uncertainty.
    """
    y_hat = y_pred[:, 0]    # Predicted magnitude
    s = y_pred[:, 1]        # Predicted log variance (uncertainty)

    # Compute loss: 0.5 * exp(-s) * (y_true - y_hat)^2 + 0.5 * s
    loss = 0.5 * torch.exp(-s) * (y_true - y_hat)**2 + 0.5 * s

    return torch.mean(loss)

# %% [markdown]
# Cell 4: Training and Evaluation Functions

# %%
#------------------------------------------------------------------------------
# Training and Evaluation Functions
#------------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, num_epochs=300, patience=5,
                run_id=None, split_num=None, model_seed=None, verbose=False):
    """
    Train the model with early stopping and learning rate scheduling.

    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Maximum number of training epochs
        patience: Patience for early stopping
        run_id: Identifier for the experimental run
        split_num: Which data split is being used (0-49)
        model_seed: Random seed used for model initialization
        verbose: Whether to print detailed progress

    Returns:
        Dictionary with training history and best model path
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=np.sqrt(0.1),
        cooldown=0, patience=4, verbose=verbose, min_lr=0.5e-6
    )

    early_stopping = EarlyStopping(
        patience=patience, verbose=verbose,
        run_id=run_id, split_num=split_num, model_seed=model_seed
    )

    criterion = custom_loss
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()

        # Calculate average losses
        val_loss /= len(val_loader)
        running_loss /= len(train_loader)

        # Learning rate scheduling and early stopping
        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if verbose:
            print(f'Epoch {epoch+1}, Loss: {running_loss:.4f}, '
                  f'Validation Loss: {val_loss:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        train_losses.append(running_loss)
        val_losses.append(val_loss)

        if early_stopping.early_stop:
            if verbose:
                print(f'Early stopping triggered at epoch {epoch+1}')
            break

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_model_path': early_stopping.best_model_path
    }

def estimate_uncertainty(model, data_loader, num_samples=50):
    """
    Estimate model uncertainty using Monte Carlo dropout.

    Args:
        model: Trained model
        data_loader: DataLoader for test data
        num_samples: Number of Monte Carlo samples

    Returns:
        Tuple of (predictions, epistemic_uncertainty, aleatoric_uncertainty, combined_uncertainty)
    """
    model.eval()

    # Enable dropout during inference for Monte Carlo sampling
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    predictions = []
    log_variances = []

    with torch.no_grad():
        for _ in range(num_samples):
            batch_predictions = []
            batch_log_variances = []
            for data, _ in data_loader:
                data = data.to(device)
                output = model(data)
                batch_predictions.append(output[:, 0].cpu().numpy())
                batch_log_variances.append(output[:, 1].cpu().numpy())
            predictions.append(np.concatenate(batch_predictions))
            log_variances.append(np.concatenate(batch_log_variances))

    predictions = np.array(predictions)
    log_variances = np.array(log_variances)

    # Calculate mean prediction
    mean_prediction = np.mean(predictions, axis=0)

    # Calculate mean of squared predictions
    yhat_squared_mean = np.mean(np.square(predictions), axis=0)

    # Calculate aleatoric uncertainty from log variances
    aleatoric_uncertainty = np.mean(np.exp(log_variances), axis=0)

    # Calculate epistemic uncertainty as standard deviation of predictions
    epistemic_uncertainty = np.std(predictions, axis=0)

    # Calculate combined uncertainty
    combined_uncertainty = yhat_squared_mean - np.square(mean_prediction) + aleatoric_uncertainty

    return mean_prediction, epistemic_uncertainty, aleatoric_uncertainty, combined_uncertainty

def evaluate_model(model_path, test_loader):
    """
    Evaluate a trained model on test data.

    Args:
        model_path: Path to the saved model weights
        test_loader: DataLoader for test data

    Returns:
        Dictionary with evaluation metrics
    """
    model = EarthquakeModel().to(device)
    model.load_state_dict(torch.load(model_path))

    # Get predictions and uncertainties
    mean_pred, epistemic_unc, aleatoric_unc, combined_unc = estimate_uncertainty(model, test_loader)

    # Get true values
    true_values = []
    for _, target in test_loader:
        true_values.append(target.numpy())
    true_values = np.concatenate(true_values)

    # Calculate MAE
    mae = np.mean(np.abs(mean_pred - true_values))

    return {
        'mae': float(mae),
        'mean_prediction': mean_pred,
        'true_values': true_values,
        'epistemic_uncertainty': epistemic_unc,
        'aleatoric_uncertainty': aleatoric_unc,
        'combined_uncertainty': combined_unc,
        'mean_epistemic_uncertainty': float(np.mean(epistemic_unc)),
        'mean_aleatoric_uncertainty': float(np.mean(aleatoric_unc)),
        'mean_combined_uncertainty': float(np.mean(combined_unc))
    }

# %% [markdown]
# Cell 5: Experimental Functions

# %%
#------------------------------------------------------------------------------
# Experimental Functions
#------------------------------------------------------------------------------

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_event_based_split(split_seed):
    """
    Create a random event-based split with the specified seed

    Args:
        split_seed: Random seed for the split

    Returns:
        Dictionary with train, val, test data and labels
    """
    # Load the data
    all_data = torch.load(all_data_file)
    all_labels = torch.load(all_labels_file)

    # Load the split information
    with open(split_info_file, 'rb') as f:
        split_info = pickle.load(f)

    unique_events = split_info['unique_events']
    event_indices = split_info['event_indices']
    train_ratio = split_info['train_ratio']
    val_ratio = split_info['val_ratio']

    # Apply the offset to get a different random seed (51-75 instead of 1-25)
    random_seed = split_seed + RANDOM_SEED_OFFSET

    # Set the seed for reproducibility using the new random seed
    print(f"  Using random seed {random_seed} for split {split_seed}")
    set_seed(random_seed)

    # Create a shuffled copy of the unique events
    events_copy = unique_events.copy()
    random.shuffle(events_copy)

    # Split events into train/val/test
    train_size = int(train_ratio * len(events_copy))
    val_size = int(val_ratio * len(events_copy))

    train_events = events_copy[:train_size]
    val_events = events_copy[train_size:train_size + val_size]
    test_events = events_copy[train_size + val_size:]

    # Collect indices for each split
    train_indices = np.concatenate([event_indices[event_id] for event_id in train_events])
    val_indices = np.concatenate([event_indices[event_id] for event_id in val_events])
    test_indices = np.concatenate([event_indices[event_id] for event_id in test_events])

    # Extract data using the indices
    train_data = all_data[train_indices]
    train_labels = all_labels[train_indices]

    val_data = all_data[val_indices]
    val_labels = all_labels[val_indices]

    test_data = all_data[test_indices]
    test_labels = all_labels[test_indices]

    return {
        'train_data': train_data,
        'train_labels': train_labels,
        'val_data': val_data,
        'val_labels': val_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'split_seed': split_seed,  # Keep the original split_seed for labeling (1-25)
        'random_seed': random_seed,  # Save the actual random seed used (51-75)
        'train_events': train_events,
        'val_events': val_events,
        'test_events': test_events
    }

def run_experiment(split_seed, model_seeds, run_id):
    """
    Run a complete experiment with multiple model initializations on a specific data split.

    Args:
        split_seed: Random seed for the split
        model_seeds: List of random seeds for model initialization
        run_id: Identifier for this experiment run

    Returns:
        Dictionary with experiment results
    """
    print(f"Running experiment with split seed {split_seed}")

    # Create the data split
    split_data = create_event_based_split(split_seed)

    # Create datasets
    train_dataset = EarthquakeDataset(split_data['train_data'], split_data['train_labels'])
    val_dataset = EarthquakeDataset(split_data['val_data'], split_data['val_labels'])
    test_dataset = EarthquakeDataset(split_data['test_data'], split_data['test_labels'])

    # Create dataloaders (UPDATED BATCH SIZE FOR LARGER INSTANCE DATASET)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

    # Log split sizes
    print(f"  Train: {len(train_dataset)} samples from {len(split_data['train_events'])} events")
    print(f"  Validation: {len(val_dataset)} samples from {len(split_data['val_events'])} events")
    print(f"  Test: {len(test_dataset)} samples from {len(split_data['test_events'])} events")

    # Run experiments with multiple random initializations
    seed_results = []

    for model_seed in model_seeds:
        print(f"  Training with model seed {model_seed}")

        # Set random seed for model initialization
        set_seed(model_seed)

        # Initialize the model
        model = EarthquakeModel().to(device)

        # Train the model
        training_result = train_model(
            model, train_loader, val_loader,
            run_id=run_id, split_num=split_seed, model_seed=model_seed,
            verbose=False  # Set to True for detailed progress
        )

        # Evaluate the model
        best_model_path = training_result['best_model_path']
        evaluation_result = evaluate_model(best_model_path, test_loader)

        # Store results
        seed_results.append({
            'model_seed': model_seed,
            'training_history': {
                'train_losses': training_result['train_losses'],
                'val_losses': training_result['val_losses']
            },
            'evaluation': evaluation_result
        })

        print(f"  Seed {model_seed} - MAE: {evaluation_result['mae']:.4f}")

    # Find median performance
    sorted_results = sorted(seed_results, key=lambda x: x['evaluation']['mae'])
    median_result = sorted_results[len(model_seeds) // 2]

    return {
        'split_seed': split_seed,
        'random_seed_used': split_seed + RANDOM_SEED_OFFSET,  # Save the actual random seed used
        'all_seed_results': seed_results,
        'median_mae': median_result['evaluation']['mae'],
        'median_model_seed': median_result['model_seed'],
        'median_aleatoric_uncertainty': median_result['evaluation']['mean_aleatoric_uncertainty'],
        'median_epistemic_uncertainty': median_result['evaluation']['mean_epistemic_uncertainty'],
        'median_combined_uncertainty': median_result['evaluation']['mean_combined_uncertainty'],
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'train_events': len(split_data['train_events']),
        'val_events': len(split_data['val_events']),
        'test_events': len(split_data['test_events'])
    }

# %% [markdown]
# Cell 6: Main Execution

# %%
#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------

if __name__ == "__main__":
    # Define model initialization seeds (these stay fixed across all experiments)
    model_seeds = [42, 123, 256, 789, 1024]  # 5 different model initializations

    # Define the specific split seeds for this notebook
    split_seeds = list(range(START_SEED, END_SEED + 1))

    # Define results file for this range of experiments
    results_file = os.path.join(output_dir, f"results_{START_SEED}_to_{END_SEED}.json")

    # Run experiments with the specified split seeds
    all_results = []

    print(f"Starting INSTANCE Event-Based Splitting Experiments {START_SEED}-{END_SEED}")
    print(f"Expected: INSTANCE (avg 10.64 seismograms/event) should show MORE pronounced")
    print(f"differences than STEAD (avg 2.14 seismograms/event) between splitting methods")
    print("-" * 80)

    for i, split_seed in enumerate(tqdm(split_seeds, desc=f"Running experiments {START_SEED}-{END_SEED}")):
        # Calculate the global run ID (to maintain consistent naming with the original code)
        global_run_id = split_seed  # This keeps the same run_id as in the original code

        # Run experiment for this split
        result = run_experiment(split_seed, model_seeds, global_run_id)
        all_results.append(result)

        # Save results after each split
        with open(results_file, 'w') as f:
            # Convert numpy arrays to Python lists before serialization
            serializable_results = numpy_to_python(all_results)
            json.dump(serializable_results, f, indent=4)

        print(f"Completed experiment for split seed {split_seed} (using random seed {split_seed + RANDOM_SEED_OFFSET})")
        print(f"Median MAE: {result['median_mae']:.4f}")
        print(f"Median Aleatoric Uncertainty: {result['median_aleatoric_uncertainty']:.4f}")
        print(f"Median Epistemic Uncertainty: {result['median_epistemic_uncertainty']:.4f}")
        print(f"Median Combined Uncertainty: {result['median_combined_uncertainty']:.4f}")
        print("-" * 50)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time/60:.2f} minutes")

    print(f"\nINSTANCE Experiment batch {START_SEED}-{END_SEED} completed. Results saved in:")
    print(f"- {results_file}")


