from open3d.ml.torch.datasets import Semantic3D
import numpy as np
from pathlib import Path
import glob
from os.path import exists

class Semantic3DForEval(Semantic3D):
    """ Semantic3D dataset wrapper for evaluation with val set assigned to test set.
    """

    def __init__(self,
                 dataset_path,
                 name='Semantic3D',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 num_points=65536,
                 class_weights=[
                     5181602, 5012952, 6830086, 1311528, 10476365, 946982,
                     334860, 269353
                 ],
                 ignored_label_inds=[0],
                 val_files=[
                     'bildstein_station3_xyz_intensity_rgb',
                     'sg27_station2_intensity_rgb'
                 ],
                 test_result_folder='./test',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (Semantic3D in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            num_points: The maximum number of points to use when splitting the dataset.
            class_weights: The class weights to use in the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            val_files: The files with the data.
            test_result_folder: The folder where the test results should be stored.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         num_points=num_points,
                         ignored_label_inds=ignored_label_inds,
                         val_files=val_files,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([0])

        self.all_files = glob.glob(str(Path(self.cfg.dataset_path) / '*.txt'))

        self.train_files = [
            f for f in self.all_files if exists(
                str(Path(f).parent / Path(f).name.replace('.txt', '.labels')))
        ]

        self.train_files = np.sort(self.train_files)
        self.val_files = []
        self.test_files = []

        for i, file_path in enumerate(self.train_files):
            for val_file in cfg.val_files:
                if val_file in file_path:
                    self.val_files.append(file_path)
                    self.test_files.append(file_path)
                    break

        self.train_files = np.sort(
            [f for f in self.train_files if f not in self.val_files])