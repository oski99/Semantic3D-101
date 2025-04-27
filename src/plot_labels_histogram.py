import os
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import numpy as np

# Label to names mapping
label_to_names = {
    0: 'unlabeled',
    1: 'man-made terrain',
    2: 'natural terrain',
    3: 'high vegetation',
    4: 'low vegetation',
    5: 'buildings',
    6: 'hard scape',
    7: 'scanning artefacts',
    8: 'cars'
}

# Get the fixed order of labels from the mapping
label_order = sorted(label_to_names.keys())
label_names = [label_to_names[label] for label in label_order]

def get_station_name(filename):
    """Extract station name from filename (removing part numbers)"""
    parts = filename.split('_part_')
    if len(parts) > 1:
        return parts[0]
    return filename.split('.')[0]

def plot_histogram(counts, title, output_path, total_points=None):
    """Generic function to plot and save a histogram"""
    plt.figure(figsize=(12, 6))
    bars = plt.bar(label_names, counts)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}' if height >= 1000 else f'{int(height)}',
                 ha='center', va='bottom')

    title_str = title
    if total_points:
        title_str += f'\nTotal points: {total_points:,}'
    plt.title(title_str)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')

    # Set y-axis to start at 0 and add some headroom
    max_count = max(counts) if max(counts) > 0 else 1
    plt.ylim(0, max_count * 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved histogram to {output_path}")

def process_directory(input_dir, output_dir=None):
    """Process all .labels files in a directory"""
    if output_dir is None:
        output_dir = os.path.join(input_dir, "label_histograms")
    os.makedirs(output_dir, exist_ok=True)

    # Collect all labels by station and for the entire dataset
    station_labels = defaultdict(list)
    all_labels = []

    for filename in os.listdir(input_dir):
        if filename.endswith('.labels'):
            file_path = os.path.join(input_dir, filename)
            station_name = get_station_name(filename)

            # Read labels from file
            with open(file_path, 'r') as f:
                labels = [int(line.strip()) for line in f.readlines() if line.strip()]

            station_labels[station_name].extend(labels)
            all_labels.extend(labels)

    # 1. Plot histograms for each station
    for station_name, labels in station_labels.items():
        if not labels:
            print(f"No labels found for {station_name}")
            continue

        # Count label occurrences
        label_counts = Counter(labels)
        counts = [label_counts.get(label, 0) for label in label_order]

        output_path = os.path.join(output_dir, f"{station_name}_histogram.png")
        plot_histogram(counts, f'Label Distribution: {station_name}',
                      output_path, len(labels))

    # 2. Plot combined histogram for entire dataset
    if all_labels:
        total_counts = Counter(all_labels)
        combined_counts = [total_counts.get(label, 0) for label in label_order]

        output_path = os.path.join(output_dir, "00_combined_dataset_histogram.png")
        plot_histogram(combined_counts, 'Label Distribution: Entire Dataset',
                      output_path, len(all_labels))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Plot label histograms for .labels files')
    parser.add_argument('input_dir', help='Directory containing .labels files')
    parser.add_argument('--output_dir', help='Directory to save histograms (default: input_dir/label_histograms)')

    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
    print("Finished processing all files")