import os
import numpy as np
import open3d as o3d

# Label to color mapping (same as Semantic3D)
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

def read_point_cloud(txt_file):
    """Read XYZ points from Semantic3D txt file"""
    points = np.loadtxt(txt_file)
    xyz = points[:, :3]
    return xyz

def read_labels(labels_file):
    """Read labels"""
    return np.loadtxt(labels_file, dtype=np.int32)

def create_colored_pcd(xyz, labels):
    """Create a colored point cloud based on labels"""
    colors = np.zeros((len(labels), 3))
    for label, color in label_to_color.items():
        colors[labels == label] = np.array(color) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def process_predictions(txt_dir, labels_dir, output_dir):
    """For each prediction label file, create colored PLY"""

    os.makedirs(output_dir, exist_ok=True)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.labels')]

    for label_file in label_files:
        base_name = label_file.replace('.labels', '')

        txt_path = os.path.join(txt_dir, f"{base_name}.txt")
        labels_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(txt_path):
            print(f"TXT file {txt_path} not found, skipping...")
            continue

        # Load data
        xyz = read_point_cloud(txt_path)
        labels = read_labels(labels_path)

        assert len(xyz) == len(labels), f"Mismatch: {base_name} (points: {len(xyz)}, labels: {len(labels)})"

        # Create colored point cloud
        pcd = create_colored_pcd(xyz, labels)

        # Save
        output_file = os.path.join(output_dir, f"{base_name}_pred.ply")
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"Saved prediction PLY: {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Color predicted Semantic3D labels and save as PLY')
    parser.add_argument('--txt_dir', help='Directory containing original .txt clouds', default='Semantic3D/processed')
    parser.add_argument('--labels_dir', help='Directory containing .labels prediction files', default='test/Semantic3D')
    parser.add_argument('--output_dir', help='Directory to save colored prediction PLY files', default='test/Semantic3D/clouds')

    args = parser.parse_args()

    process_predictions(args.txt_dir, args.labels_dir, args.output_dir)
    print("All prediction PLYs created.")
