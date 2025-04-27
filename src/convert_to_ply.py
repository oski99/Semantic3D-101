import os
import numpy as np
import open3d as o3d

# Label to color mapping (Semantic3D colors)
label_to_color = {
    0: [0, 0, 0],        # unlabeled - black
    1: [70, 70, 70],     # man-made terrain - dark gray
    2: [0, 255, 0],      # natural terrain - green
    3: [0, 100, 0],      # high vegetation - dark green
    4: [152, 251, 152],  # low vegetation - light green
    5: [255, 0, 0],      # buildings - red
    6: [200, 200, 200],  # hard scape - light gray
    7: [255, 255, 0],    # scanning artefacts - yellow
    8: [0, 0, 255]       # cars - blue
}

def read_semantic3d_file(txt_file, labels_file):
    """Read point cloud data and labels from Semantic3D files"""
    # Read points (x,y,z,intensity,r,g,b)
    points = np.loadtxt(txt_file)
    xyz = points[:, :3]
    if points.shape[1] > 3:  # if intensity and rgb exist
        intensity = points[:, 3]
        rgb = points[:, 4:7] if points.shape[1] >= 7 else None
    else:
        intensity = None
        rgb = None

    # Read labels
    labels = np.loadtxt(labels_file, dtype=np.int32)

    return xyz, intensity, rgb, labels

def create_point_cloud(xyz, colors):
    """Create Open3D point cloud with given colors"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def convert_to_ply(txt_file, labels_file, rgb_output_dir, label_output_dir):
    """Convert Semantic3D files to both RGB and label-colored PLY"""
    # Read files
    xyz, intensity, rgb, labels = read_semantic3d_file(txt_file, labels_file)

    # Create base name for output files
    base_name = os.path.basename(txt_file).replace('.txt', '')

    # 1. Save RGB version (if RGB data exists)
    if rgb is not None:
        rgb_colors = rgb / 255.0
        rgb_pcd = create_point_cloud(xyz, rgb_colors)
        rgb_output_file = os.path.join(rgb_output_dir, f"{base_name}.ply")
        o3d.io.write_point_cloud(rgb_output_file, rgb_pcd)
        print(f"Saved RGB version to {rgb_output_file}")

    # 2. Save label-colored version
    label_colors = np.zeros((len(labels), 3))
    for label, color in label_to_color.items():
        label_colors[labels == label] = np.array(color) / 255.0

    label_pcd = create_point_cloud(xyz, label_colors)
    label_output_file = os.path.join(label_output_dir, f"{base_name}_labels.ply")
    o3d.io.write_point_cloud(label_output_file, label_pcd)
    print(f"Saved label version to {label_output_file}")

def process_directory(input_dir, rgb_output_dir, label_output_dir):
    """Process all Semantic3D files in directory"""
    os.makedirs(rgb_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    # Find all .txt files that have corresponding .labels files
    txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]

    for txt_file in txt_files:
        base_name = txt_file.replace('.txt', '')
        labels_file = f"{base_name}.labels"

        if os.path.exists(os.path.join(input_dir, labels_file)):
            txt_path = os.path.join(input_dir, txt_file)
            labels_path = os.path.join(input_dir, labels_file)

            convert_to_ply(txt_path, labels_path, rgb_output_dir, label_output_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert Semantic3D dataset to colored PLY files')
    parser.add_argument('input_dir', help='Directory containing Semantic3D .txt and .labels files')
    parser.add_argument('--rgb_output_dir',
                      help='Directory to save RGB PLY files (default: input_dir/ply_rgb)',
                      default='ply_rgb')
    parser.add_argument('--label_output_dir',
                      help='Directory to save label-colored PLY files (default: input_dir/ply_labels)',
                      default='ply_labels')

    args = parser.parse_args()

    # Convert relative output directories to absolute paths
    if not os.path.isabs(args.rgb_output_dir):
        args.rgb_output_dir = os.path.join(args.input_dir, args.rgb_output_dir)
    if not os.path.isabs(args.label_output_dir):
        args.label_output_dir = os.path.join(args.input_dir, args.label_output_dir)

    process_directory(args.input_dir, args.rgb_output_dir, args.label_output_dir)
    print(f"Conversion complete.")
    print(f"RGB versions saved to: {args.rgb_output_dir}")
    print(f"Label-colored versions saved to: {args.label_output_dir}")