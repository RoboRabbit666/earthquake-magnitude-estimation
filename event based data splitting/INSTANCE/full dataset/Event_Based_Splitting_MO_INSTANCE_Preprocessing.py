# -*- coding: utf-8 -*-
"""Event_Based_Splitting_MO_INSTANCE_Preprocessing.ipynb

Cell 1: Initial Setup with SeisBench Download
"""

# -*- coding: utf-8 -*-
"""
This notebook processes the INSTANCE dataset to create event-based groupings
and prepares data files for 50 different random event-based splits.
MODIFIED: Uses SeisBench to download INSTANCE dataset directly to avoid Google Drive quota issues.
"""

# Install and import required packages
!pip install seisbench
!pip install torch torchvision torchaudio
!pip install h5py pandas tqdm matplotlib torchinfo seaborn

from google.colab import drive
import time
import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import torch
from collections import defaultdict
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal
import seisbench.data as sbd

# Start time
start_time = time.time()

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Mount Google Drive for output directory only
drive.mount('/content/drive')

# Set SeisBench to use local VM storage (not Google Drive) to avoid quota issues
os.environ['SEISBENCH_CACHE_ROOT'] = '/content/.seisbench'
print(f"SeisBench cache root: {os.environ['SEISBENCH_CACHE_ROOT']}")

# Download INSTANCE dataset using SeisBench
print("Downloading INSTANCE dataset using SeisBench...")
print("This will download to local VM storage, avoiding Google Drive quota issues...")
instance = sbd.InstanceCounts()

# # Define the paths to local data files
# metadata_path = "/content/drive/My Drive/INSTANCE_Dataset/metadata.csv"
# hdf5_path = "/content/drive/My Drive/INSTANCE_Dataset/waveforms.hdf5"

# Get paths to the downloaded files
instance_path = instance.path
metadata_path = os.path.join(instance_path, "metadata.csv")
hdf5_path = os.path.join(instance_path, "waveforms.hdf5")

# Output directory (still on Google Drive for results)
output_dir = "/content/drive/My Drive/2023-2024/UCL MSc in DSML/Term 3/MSc Project/Code/INSTANCE_Event_Based"
os.makedirs(output_dir, exist_ok=True)

# Verify files exist
print(f"INSTANCE dataset path: {instance_path}")
print(f"Files in INSTANCE dataset folder:")
print(os.listdir(instance_path))

assert os.path.isfile(metadata_path), f"Metadata file not found at {metadata_path}"
assert os.path.isfile(hdf5_path), f"HDF5 file not found at {hdf5_path}"

print(f"Using SeisBench INSTANCE files:")
print(f"Metadata: {metadata_path}")
print(f"HDF5: {hdf5_path}")
print(f"Output directory: {output_dir}")

"""Cell 2: Data Loading and Initial Filtering"""

"""Cell 2: Data Loading and Initial Filtering"""

# Load and access the metadata from the CSV file
metadata = pd.read_csv(metadata_path, low_memory=False)
print(f"Initial number of metadata entries: {len(metadata)}")

# Ensure the 'source_origin_time' column is in datetime format
metadata['source_origin_time'] = pd.to_datetime(metadata['source_origin_time'])

# Sort by source_origin_time to ensure chronological order
metadata = metadata.sort_values(by='source_origin_time')

# # Filter metadata for dates from September 2018 onwards
# from datetime import datetime, timezone
# start_date = datetime(2018, 9, 1, tzinfo=timezone.utc)
# metadata = metadata[metadata['source_origin_time'] >= start_date]
# print(f"Number of metadata entries from 1st September 2018 to 31st January 2020: {len(metadata)}")

# Plot the histogram of the earthquake magnitudes before filtering
plt.figure(figsize=(10, 6))
sns.histplot(metadata["source_magnitude"], bins=30, kde=True)
plt.xlabel('Magnitude', fontweight='bold', fontsize=14)
plt.ylabel('Number', fontweight='bold', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
max_magnitude = metadata["source_magnitude"].max()
min_magnitude = metadata["source_magnitude"].min()
plt.text(0.7*max_magnitude, plt.gca().get_ylim()[1]*0.8,
         f'Max: {max_magnitude:.2f} M\nMin: {min_magnitude:.2f} M',
         bbox=dict(facecolor='none', edgecolor='red', boxstyle='round,pad=1'), fontsize=14)
plt.tight_layout()
plt.show()

print(f"Number of metadata entries now: {len(metadata)}")

"""Cell 3: Apply Data Filtering Criteria"""

"""Cell 3: Apply Data Filtering Criteria"""

import copy

# Create a deep copy of the metadata
filtered_metadata = copy.deepcopy(metadata)

# Set sampling rate
filtered_metadata['trace_sampling_rate_hz'] = 100

# Adjusted filtering for INSTANCE dataset with 30s window
filters = [
    ('path_ep_distance_km <= 110', filtered_metadata.path_ep_distance_km <= 110),
    ('source_magnitude_type == ML', filtered_metadata.source_magnitude_type == 'ML'),
    ('trace_P_arrival_sample.notnull()', filtered_metadata.trace_P_arrival_sample.notnull()),
    ('path_travel_time_P_s.notnull()', filtered_metadata.path_travel_time_P_s.notnull()),
    ('path_travel_time_P_s > 0', filtered_metadata.path_travel_time_P_s > 0),
    ('path_ep_distance_km.notnull()', filtered_metadata.path_ep_distance_km.notnull()),
    ('path_ep_distance_km > 0', filtered_metadata.path_ep_distance_km > 0),
    ('source_depth_km.notnull()', filtered_metadata.source_depth_km.notnull()),
    ('source_magnitude.notnull()', filtered_metadata.source_magnitude.notnull()),
    ('path_backazimuth_deg.notnull()', filtered_metadata.path_backazimuth_deg.notnull()),
    ('path_backazimuth_deg > 0', filtered_metadata.path_backazimuth_deg > 0),
]

# Apply filters one by one and keep track of the data
for desc, filt in filters:
    filtered_metadata = filtered_metadata[filt]
    print(f"After filter '{desc}': {len(filtered_metadata)} entries")

# Ensure at least 30 seconds of data after P-arrival
filtered_metadata = filtered_metadata[filtered_metadata.trace_P_arrival_sample + filtered_metadata.trace_sampling_rate_hz * 30 <= filtered_metadata.trace_npts]
print(f"After ensuring 30s after P arrival: {len(filtered_metadata)} entries")

# Calculate time window statistics
filtered_metadata['window_start'] = filtered_metadata.trace_P_arrival_sample - filtered_metadata.trace_sampling_rate_hz * 5  # 5 seconds before P arrival
filtered_metadata['window_end'] = filtered_metadata.trace_P_arrival_sample + filtered_metadata.trace_sampling_rate_hz * 25  # 25 seconds after P arrival (total 30s window)

# Ensure window start is not negative
filtered_metadata = filtered_metadata[filtered_metadata.window_start >= 0]
print(f"After ensuring non-negative window start: {len(filtered_metadata)} entries")

print(f"\nFinal number of filtered metadata entries: {len(filtered_metadata)}")

"""Cell 4: Apply SNR Filter and Multi-Observations"""

"""Cell 4: Apply SNR Filter and Multi-Observations"""

# Process SNR
# Check if 'snr_db' column exists, if not, use appropriate column names from INSTANCE dataset
if 'snr_db' in filtered_metadata.columns:
    def string_convertor(dd):
        if isinstance(dd, str):
            dd2 = dd.split()
            SNR = []
            for d in dd2:
                if d not in ['[', ']']:
                    dL = d.split('[')
                    dR = d.split(']')
                    if len(dL) == 2:
                        dig = dL[1]
                    elif len(dR) == 2:
                        dig = dR[0]
                    elif len(dR) == 1 and len(dL) == 1:
                        dig = d
                    try:
                        dig = float(dig)
                    except Exception:
                        dig = None
                    SNR.append(dig)
            return np.mean([x for x in SNR if x is not None])
        else:
            return np.nan

    filtered_metadata['snr_db'] = filtered_metadata.snr_db.apply(string_convertor)
    filtered_metadata = filtered_metadata[filtered_metadata.snr_db >= 20]
else:
    # Use appropriate SNR columns from INSTANCE dataset
    snr_columns = ['trace_E_snr_db', 'trace_N_snr_db', 'trace_Z_snr_db']
    filtered_metadata['avg_snr_db'] = filtered_metadata[snr_columns].mean(axis=1)
    filtered_metadata = filtered_metadata[filtered_metadata.avg_snr_db >= 20]

print(f"Number of records after SNR filtering: {len(filtered_metadata)}")

# Plot the histogram of the earthquake magnitudes after filtering
plt.figure(figsize=(10, 6))
sns.histplot(filtered_metadata["source_magnitude"], bins=30, kde=True, color='brown')
plt.xlabel('Magnitude', fontweight='bold', fontsize=14)
plt.ylabel('Number', fontweight='bold', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True)
max_magnitude = filtered_metadata["source_magnitude"].max()
min_magnitude = filtered_metadata["source_magnitude"].min()
plt.text(0.7*max_magnitude, plt.gca().get_ylim()[1]*0.8,
         f'Max: {max_magnitude:.2f} M\nMin: {min_magnitude:.2f} M',
         bbox=dict(facecolor='none', edgecolor='red', boxstyle='round,pad=1'), fontsize=14)
plt.tight_layout()
plt.show()

# Implementing multi-observations with a threshold of >= 400
print("Filtering for multi-observations stations...")
uniq_ins = filtered_metadata.station_code.unique()
labM = []

print("Counting observations per station...")
for station in tqdm(uniq_ins):
    count = sum(filtered_metadata.station_code == station)
    if count >= 400:
        labM.append(station)
        print(f"{station}: {count} observations")

print(f"Number of stations with ≥400 observations: {len(labM)}")

# Save the multi-observations stations list
multi_observations_path = os.path.join(output_dir, "multi_observations.npy")
np.save(multi_observations_path, labM)

# Filter the dataset to include only records from stations with ≥400 observations
df_filtered = filtered_metadata[filtered_metadata.station_code.isin(labM)]
print(f"Number of records after multi-observations filtering: {len(df_filtered)}")

"""Cell 5: Group Seismograms by Events (KEY EVENT-BASED LOGIC)"""

"""Cell 5: Group Seismograms by Events (KEY ADAPTATION)"""

# Group seismograms by their source event ID (THIS IS THE KEY EVENT-BASED LOGIC)
print("Grouping seismograms by source event ID...")
event_to_seismograms = defaultdict(list)
seismogram_to_event = {}
event_metadata = {}

for index, row in df_filtered.iterrows():
    # Use source_id as the event identifier (adapt this if INSTANCE uses a different field)
    event_id = row['source_id'] if 'source_id' in row else row['source_event_id']  # Adapt field name as needed
    seismogram_id = row['trace_name']

    event_to_seismograms[event_id].append(seismogram_id)
    seismogram_to_event[seismogram_id] = event_id

    # Store metadata for each event (only once per event)
    if event_id not in event_metadata:
        event_metadata[event_id] = {
            'magnitude': row['source_magnitude'],
            'latitude': row['source_latitude_deg'], # Use correct column name for latitude: source_latitude_deg
            'longitude': row['source_longitude_deg'], # Use correct column name for longitude: source_longitude_deg
            'depth': row['source_depth_km'],
            'origin_time': row['source_origin_time']
        }

print(f"Number of unique earthquake events: {len(event_to_seismograms)}")
print(f"Average seismograms per event: {len(df_filtered) / len(event_to_seismograms):.2f}")

# Calculate the frequency of events with each number of seismograms
seismograms_per_event = [len(seismograms) for seismograms in event_to_seismograms.values()]

# Print summary statistics
print("\nDistribution statistics of seismograms per event:")
print(f"Minimum: {min(seismograms_per_event)} seismograms per event")
print(f"Maximum: {max(seismograms_per_event)} seismograms per event")

# Count frequency of each number of seismograms
counts = {}
for count in seismograms_per_event:
    counts[count] = counts.get(count, 0) + 1

# Output the counts
print("\nDetailed frequency distribution:")
for num_seismograms in range(1, min(21, max(seismograms_per_event) + 1)):  # Show up to 20 for readability
    num_events = counts.get(num_seismograms, 0)
    if num_events > 0:  # Only print non-zero counts to save space
        print(f"Number of events with {num_seismograms} seismogram{'s' if num_seismograms > 1 else ''}: {num_events}")

# Analyze the distribution of seismograms per event
seismograms_per_event = [len(seismograms) for seismograms in event_to_seismograms.values()]
plt.figure(figsize=(10, 6))
plt.hist(seismograms_per_event, bins=30)
plt.xlabel('Number of Seismograms per Event', fontweight='bold', fontsize=14)
plt.ylabel('Frequency', fontweight='bold', fontsize=14)
plt.title('Distribution of Seismograms per Earthquake Event', fontweight='bold', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()

"""Cell 6: Check Available Seismograms and Load Data"""

"""Cell 6: Check Available Seismograms and Load Data"""

# Check which seismograms exist in the HDF5 file
print("Checking available seismograms in HDF5 file...")

def explore_hdf5_structure(file_path):
    """Explore the HDF5 structure to understand data organization"""
    with h5py.File(file_path, 'r') as f:
        print("HDF5 structure:")
        f.visititems(lambda name, obj: print(f"  {name}: {type(obj)}"))

explore_hdf5_structure(hdf5_path)

# Get list of valid seismograms from the filtered metadata
valid_seismogram_ids = df_filtered['trace_name'].tolist()
print(f"Number of seismograms to load: {len(valid_seismogram_ids)}")

# Rebuild the event-to-seismograms mapping with only valid seismograms
valid_event_to_seismograms = defaultdict(list)
for seis_id in valid_seismogram_ids:
    event_id = seismogram_to_event[seis_id]
    valid_event_to_seismograms[event_id].append(seis_id)

# Remove events that have no valid seismograms left
valid_event_to_seismograms = {k: v for k, v in valid_event_to_seismograms.items() if len(v) > 0}
print(f"Number of events with valid seismograms: {len(valid_event_to_seismograms)}")

# Function to load seismogram data from HDF5 file (adapted for INSTANCE)
def load_seismogram_instance(hdf5_path, seismogram_id, metadata_row):
    """
    Load a single seismogram from the INSTANCE HDF5 file

    Args:
        hdf5_path: Path to the HDF5 file
        seismogram_id: ID of the seismogram to load (trace_name)
        metadata_row: Row from metadata containing trace information

    Returns:
        Tuple of (data, magnitude) or (None, None) if seismogram not found

    Notes:
        - Data is a 30-second window: 5s before P-arrival and 25s after
        - Resampled to 100Hz if necessary
    """
    try:
        with h5py.File(hdf5_path, 'r') as hdf:
            # Parse the trace name to get bucket and trace index
            bucket, trace_info = seismogram_id.split('$')
            trace_index = int(trace_info.split(',')[0])

            # Retrieve waveform data
            waveform = np.array(hdf['data'][bucket][trace_index])

            sampling_rate = metadata_row['trace_sampling_rate_hz']
            spt = int(metadata_row['trace_P_arrival_sample'])

            # Adjust window size based on sampling rate
            window_samples = int(30 * sampling_rate)  # 30 seconds window
            start = max(0, spt - int(5 * sampling_rate))  # 5 seconds before P arrival
            end = start + window_samples

            if start >= waveform.shape[1] or end > waveform.shape[1]:
                print(f"Skipping event {seismogram_id}: Invalid window")
                return None, None

            dshort = waveform[:, start:end]

            # Ensure the shape is correct
            if dshort.shape[1] != window_samples:
                print(f"Skipping event {seismogram_id}: Incorrect window size")
                return None, None

            # Resample to 100 Hz if necessary
            if sampling_rate != 100:
                dshort = scipy.signal.resample(dshort, 3000, axis=1)

            mag = round(float(metadata_row['source_magnitude']), 2)

            return dshort, mag

    except Exception as e:
        print(f"Error processing event {seismogram_id}: {e}")
        return None, None

"""Cell 7: Load All Valid Seismograms and Create Event Data"""

"""Cell 7: Load All Valid Seismograms and Create Event Data"""

# Load all valid seismograms
print("Loading all valid seismograms...")
all_data = []
all_labels = []
all_event_ids = []  # To track which event each seismogram belongs to
all_seismogram_ids = []  # To track the original seismogram ID

# Event data dict to store waveforms by event
event_waveform_data = {}

# Create a lookup dictionary for metadata rows
metadata_lookup = {row['trace_name']: row for _, row in df_filtered.iterrows()}

for event_id, seismogram_ids in tqdm(valid_event_to_seismograms.items(), desc="Loading events"):
    event_waveforms = []

    for seis_id in seismogram_ids:
        if seis_id not in metadata_lookup:
            continue

        metadata_row = metadata_lookup[seis_id]
        data, label = load_seismogram_instance(hdf5_path, seis_id, metadata_row)

        if data is None:
            continue

        all_data.append(data)
        all_labels.append(label)
        all_event_ids.append(event_id)
        all_seismogram_ids.append(seis_id)

        # Store for event visualization
        event_waveforms.append({
            'data': data,
            'magnitude': label,
            'seismogram_id': seis_id
        })

    if event_waveforms:  # Only store events that have valid waveforms
        event_waveform_data[event_id] = event_waveforms

# Convert lists to numpy arrays
all_data = np.array(all_data)
all_labels = np.array(all_labels)
all_event_ids = np.array(all_event_ids)
all_seismogram_ids = np.array(all_seismogram_ids)

print(f"Final dataset: {len(all_data)} seismograms from {len(np.unique(all_event_ids))} events")
print(f"Data shape: {all_data.shape}, Labels shape: {all_labels.shape}")

# Save the mapping between seismogram indices and their IDs
with open(os.path.join(output_dir, 'seismogram_indices.pkl'), 'wb') as f:
    pickle.dump({
        'seismogram_ids': all_seismogram_ids,
        'event_ids': all_event_ids
    }, f)

"""Cell 8: Visualize Example Events"""

"""Cell 8: Visualize Example Events"""

# Visualize example events showing all seismograms from the same event
# Find events with multiple seismograms for visualization
events_with_multiple_seismograms = []
for event_id, waveforms in event_waveform_data.items():
    if len(waveforms) > 1:
        events_with_multiple_seismograms.append((event_id, len(waveforms)))

# Sort by number of seismograms (descending)
events_with_multiple_seismograms.sort(key=lambda x: x[1], reverse=True)

# Function to visualize all seismograms from an event
def visualize_event_seismograms_instance(event_id, max_seismograms=6):
    """
    Visualize all seismograms from a specific event to show event grouping

    Args:
        event_id: ID of the event to visualize
        max_seismograms: Maximum number of seismograms to display
    """
    waveforms = event_waveform_data[event_id]
    event_info = event_metadata.get(event_id, {})

    # Limit the number of seismograms to display
    num_seismograms = min(len(waveforms), max_seismograms)
    waveforms = waveforms[:num_seismograms]

    # Create a figure with subplots for each seismogram (3 components each)
    fig = plt.figure(figsize=(15, num_seismograms * 5))
    fig.suptitle(f"Event ID: {event_id}\nMagnitude: {event_info.get('magnitude', 'N/A')}, "
                f"Depth: {event_info.get('depth', 'N/A')} km\n"
                f"Location: ({event_info.get('latitude', 'N/A')}, {event_info.get('longitude', 'N/A')})\n"
                f"Total Seismograms: {len(event_waveform_data[event_id])}",
                fontsize=16, fontweight='bold')

    components = ['East-West', 'North-South', 'Vertical']
    time = np.arange(3000) / 100  # Convert to seconds (100Hz sampling)

    for i, waveform in enumerate(waveforms):
        data = waveform['data']
        seis_id = waveform['seismogram_id']

        for j in range(3):
            ax = fig.add_subplot(num_seismograms, 3, i*3 + j + 1)
            ax.plot(time, data[j, :], 'k')

            ax.set_title(f"Seismogram {i+1} - {components[j]}")
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Amplitude')
            ax.axvline(x=5.0, color='blue', linestyle='--', linewidth=1, label='P-arrival')
            ax.grid(True)

            # Only add legend to the first subplot
            if i == 0 and j == 0:
                ax.legend()

            # Add seismogram ID as a text annotation
            if j == 2:  # Add to the vertical component only
                ax.annotate(f"ID: {seis_id}", xy=(0.5, -0.4), xycoords='axes fraction',
                            fontsize=10, ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    plt.savefig(os.path.join(output_dir, f'event_{event_id}_seismograms.png'), dpi=150)
    plt.show()

# Visualize a few events with multiple seismograms
print("\nVisualizing example events with multiple seismograms:")
for i, (event_id, count) in enumerate(events_with_multiple_seismograms[:3]):
    print(f"Example {i+1}: Event {event_id} with {count} seismograms")
    visualize_event_seismograms_instance(event_id)

# Plot an example single seismogram
def plot_example_seismogram_instance(data, index=0):
    """Plot an example three-component seismogram"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    time = np.arange(data.shape[2]) / 100  # Convert to seconds (100Hz sampling)
    components = ['East-West', 'North-South', 'Vertical']

    for i in range(3):
        axes[i].plot(time, data[index, i, :], 'k')
        axes[i].set_ylabel(f'{components[i]}\nAmplitude', fontweight='bold', fontsize=14)
        axes[i].axvline(x=5.0, color='blue', linestyle='--', linewidth=2,
                      label='P-arrival (5s mark)')
        axes[i].grid(True)
        axes[i].legend(fontsize=12)

    axes[2].set_xlabel('Time (seconds)', fontweight='bold', fontsize=14)
    plt.suptitle(f'Example INSTANCE Seismogram (Magnitude: {all_labels[index]})',
                fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

# Plot an example seismogram
if len(all_data) > 0:
    plot_example_seismogram_instance(all_data)

"""Cell 9: Save Data and Create Split Information"""

"""Cell 9: Save Data and Create Split Information"""

# Convert to tensors
all_data_tensor = torch.tensor(all_data, dtype=torch.float32)
all_labels_tensor = torch.tensor(all_labels, dtype=torch.float32)

# Save the entire dataset and associated event IDs
print("Saving data files...")
torch.save(all_data_tensor, os.path.join(output_dir, 'all_data.pt'))
torch.save(all_labels_tensor, os.path.join(output_dir, 'all_labels.pt'))

# Save the mapping between indices and event IDs for future reference
with open(os.path.join(output_dir, 'index_to_event_id.pkl'), 'wb') as f:
    pickle.dump({
        'all_event_ids': all_event_ids,
        'event_to_seismograms': valid_event_to_seismograms,
        'event_metadata': event_metadata
    }, f)

# Create function to generate the splits file without using seeds
# This file will be used by the second notebook which will apply its own seeds
def generate_event_based_split_indices(all_event_ids, train_ratio=0.7, val_ratio=0.1):
    """
    Generate the splits file without applying any shuffling
    The second notebook will handle the randomization with specific seeds

    Args:
        all_event_ids: Array of event IDs corresponding to each seismogram
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set

    Returns:
        Dictionary with unique events and their corresponding seismogram indices
    """
    # Get unique event IDs (maintaining order)
    unique_events = []
    for event_id in all_event_ids:
        if event_id not in unique_events:
            unique_events.append(event_id)

    unique_events = np.array(unique_events)
    print(f"Found {len(unique_events)} unique events for split preparation")

    # Get the indices for each event
    event_indices = {}
    for event_id in unique_events:
        indices = np.where(all_event_ids == event_id)[0]
        event_indices[event_id] = indices

    return {
        'unique_events': unique_events,
        'event_indices': event_indices,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio
    }

# Generate the splits file (without applying randomization)
print("Preparing split information for second notebook...")
split_info = generate_event_based_split_indices(all_event_ids)

# Save the split information
with open(os.path.join(output_dir, 'event_split_info.pkl'), 'wb') as f:
    pickle.dump(split_info, f)

"""Cell 10: Summary and Final Statistics"""

"""Cell 10: Summary and Final Statistics"""

# Print a summary of what was created
print("\nFiles created in the output directory:")
print(f"1. all_data.pt - All seismogram data tensors")
print(f"2. all_labels.pt - All magnitude labels")
print(f"3. index_to_event_id.pkl - Mapping between indices and event IDs")
print(f"4. event_split_info.pkl - Information for creating random event-based splits")
print(f"5. multi_observations.npy - List of stations with ≥400 observations")
print(f"6. seismogram_indices.pkl - Mapping between indices and seismogram IDs")
print(f"7. Event visualization images (PNG files)")

# Print statistics about the event distribution
seismograms_per_event = [len(event_waveform_data[event_id]) for event_id in event_waveform_data.keys()]
print(f"\nEvent Distribution Statistics:")
print(f"Total events: {len(event_waveform_data)}")
print(f"Total seismograms: {len(all_data)}")
print(f"Average seismograms per event: {np.mean(seismograms_per_event):.2f}")
print(f"Median seismograms per event: {np.median(seismograms_per_event):.0f}")
print(f"Max seismograms per event: {max(seismograms_per_event)}")
print(f"Min seismograms per event: {min(seismograms_per_event)}")

# Print example events for reference
print("\nExample events with multiple seismograms:")
for i, (event_id, count) in enumerate(events_with_multiple_seismograms[:10]):
    print(f"Event {event_id}: {count} seismograms")

# Report execution time
end_time = time.time()
print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")

print("\nPreprocessing complete! Now you can run the 50 experiments notebook using the same logic as STEAD.")
print("\nKey differences from STEAD:")
print(f"- INSTANCE average seismograms per event: {np.mean(seismograms_per_event):.2f}")
print(f"- STEAD average seismograms per event: 2.14")
print(f"- INSTANCE may show more pronounced differences between event-based and seismogram-based splitting")

print("\nINSTANCE event-based preprocessing completed successfully!")
print("Used SeisBench download - no Google Drive quota issues!")