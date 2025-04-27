import logging
import numpy as np
import pandas as pd
import os, glob
import argparse
from pathlib import Path
from os.path import join, exists
from tqdm import tqdm
from open3d.ml.datasets import utils

def parse_args():
    parser = argparse.ArgumentParser(
        description='Split large pointclouds in Semantic3D.')
    parser.add_argument('--dataset_path',
                        help='path to Semantic3D',
                        required=True)
    parser.add_argument('--out_path', help='Output path', default=None)
    parser.add_argument(
        '--size_limit',
        help='Maximum size of processed pointcloud in Megabytes.',
        default=2000,
        type=int)

    args = parser.parse_args()

    dict_args = vars(args)
    for k in dict_args:
        v = dict_args[k]
        print("{}: {}".format(k, v) if v is not None else "{} not given".
              format(k))

    return args

def process_small_file(input_path, output_path, sub_grid_size):
    """Process a small file that fits in memory."""
    pc = pd.read_csv(input_path,
                     header=None,
                     delim_whitespace=True,
                     dtype=np.float32).values

    labels = pd.read_csv(input_path.replace(".txt", ".labels"),
                         header=None,
                         delim_whitespace=True,
                         dtype=np.int32).values
    labels = np.array(labels, dtype=np.int32).reshape((-1,))

    points, feat, labels = utils.DataProcessing.grid_subsampling(
        pc[:, :3],
        features=pc[:, 3:],
        labels=labels,
        grid_size=sub_grid_size)
    pc = np.concatenate([points, feat], 1)

    np.savetxt(output_path, pc, fmt='%.3f %.3f %.3f %i %i %i %i')
    np.savetxt(output_path.replace('.txt', '.labels'), labels, fmt='%i')

import itertools

def process_large_file(input_path, output_prefix, parts, sub_grid_size):
    """Process a large file by splitting it into parts and processing each part separately."""
    # First determine the split points by scanning through the file once
    with open(input_path, 'r') as f:
        total_rows = sum(1 for _ in f)

    rows_per_part = total_rows // parts
    split_points = [i * rows_per_part for i in range(parts + 1)]
    split_points[-1] = total_rows  # Ensure last part includes all remaining rows

    # Process each part
    for part_num in range(parts):
        start_row = split_points[part_num]
        end_row = split_points[part_num + 1]

        # Read the chunk for this part using islice
        with open(input_path, 'r') as pc_file:
            pc_chunk = [list(map(float, line.strip().split()))
                        for line in itertools.islice(pc_file, start_row, end_row)]

        # Read corresponding labels using islice
        labels_path = input_path.replace(".txt", ".labels")
        with open(labels_path, 'r') as labels_file:
            labels_chunk = [int(line.strip())
                          for line in itertools.islice(labels_file, start_row, end_row)]

        # Convert to numpy arrays
        pc_chunk = np.array(pc_chunk, dtype=np.float32)
        labels_chunk = np.array(labels_chunk, dtype=np.int32)

        # Process this chunk
        points, feat, labels = utils.DataProcessing.grid_subsampling(
            pc_chunk[:, :3],
            features=pc_chunk[:, 3:],
            labels=labels_chunk,
            grid_size=sub_grid_size)
        processed_pc = np.concatenate([points, feat], 1)

        # Save this part
        output_path = f"{output_prefix}_part_{part_num}.txt"
        np.savetxt(output_path, processed_pc, fmt='%.3f %.3f %.3f %i %i %i %i')
        np.savetxt(output_path.replace('.txt', '.labels'), labels, fmt='%i')

# def process_large_file(input_path, output_prefix, parts, sub_grid_size):
#     """Process a large file by splitting it into parts and processing each part separately."""
#     # First determine the split points by scanning through the file once
#     total_rows = 0
#     with open(input_path, 'r') as f:
#         for _ in f:
#             total_rows += 1

#     rows_per_part = total_rows // parts
#     split_points = [i * rows_per_part for i in range(parts + 1)]
#     split_points[-1] = total_rows  # Ensure last part includes all remaining rows

#     # Process each part
#     for part_num in range(parts):
#         start_row = split_points[part_num]
#         end_row = split_points[part_num + 1]

#         # Read the chunk for this part
#         pc_chunk = []
#         labels_chunk = []
#         current_row = 0

#         # Read point cloud data
#         with open(input_path, 'r') as pc_file:
#             for line in pc_file:
#                 if current_row >= start_row and current_row < end_row:
#                     values = list(map(float, line.strip().split()))
#                     pc_chunk.append(values)
#                 current_row += 1
#                 if current_row >= end_row:
#                     break

#         # Read corresponding labels
#         labels_path = input_path.replace(".txt", ".labels")
#         current_row = 0
#         with open(labels_path, 'r') as labels_file:
#             for line in labels_file:
#                 if current_row >= start_row and current_row < end_row:
#                     label = int(line.strip())
#                     labels_chunk.append(label)
#                 current_row += 1
#                 if current_row >= end_row:
#                     break

#         # Convert to numpy arrays
#         pc_chunk = np.array(pc_chunk, dtype=np.float32)
#         labels_chunk = np.array(labels_chunk, dtype=np.int32)

#         # Process this chunk
#         points, feat, labels = utils.DataProcessing.grid_subsampling(
#             pc_chunk[:, :3],
#             features=pc_chunk[:, 3:],
#             labels=labels_chunk,
#             grid_size=sub_grid_size)
#         processed_pc = np.concatenate([points, feat], 1)

#         # Save this part
#         output_path = f"{output_prefix}_part_{part_num}.txt"
#         np.savetxt(output_path, processed_pc, fmt='%.3f %.3f %.3f %i %i %i %i')
#         np.savetxt(output_path.replace('.txt', '.labels'), labels, fmt='%i')

def preprocess(args):
    """Main preprocessing function."""
    dataset_path = args.dataset_path
    out_path = args.out_path
    size_limit = args.size_limit  # Size in megabytes
    sub_grid_size = 0.01

    if out_path is None:
        out_path = Path(dataset_path) / 'processed'
        print("out_path not given, Saving output in {}".format(out_path))

    all_files = glob.glob(str(Path(dataset_path) / '*.txt'))
    train_files = [
        f for f in all_files
        if exists(str(Path(f).parent / Path(f).name.replace('.txt', '.labels')))
    ]

    # Sort files by size (largest first)
    train_files.sort(key=lambda x: Path(x).stat().st_size, reverse=False)

    os.makedirs(out_path, exist_ok=True)

    for file_path in tqdm(train_files):
        file_size = Path(file_path).stat().st_size / 1e6  # Size in MB

        if file_size <= size_limit:
            # Small file - process normally
            output_path = join(out_path, Path(file_path).name)
            process_small_file(file_path, output_path, sub_grid_size)
        else:
            # Large file - split into parts
            parts = int(file_size / size_limit) + 1
            print(f"Splitting {Path(file_path).name} into {parts} parts")

            output_prefix = join(out_path, Path(file_path).name.replace('.txt', ''))
            process_large_file(file_path, output_prefix, parts, sub_grid_size)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
    )

    args = parse_args()
    preprocess(args)