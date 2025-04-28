import os
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import logging
from src.semantic3d_wrapper import Semantic3DForEval

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)

framework = "torch" # or tf
cfg_file = "configs/randlanet_semantic3d.yml"
cfg = _ml3d.utils.Config.load_from_file(cfg_file)

Pipeline = _ml3d.utils.get_module("pipeline", cfg.pipeline.name, framework)
Model = _ml3d.utils.get_module("model", cfg.model.name, framework)

dataset = Semantic3DForEval(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
model = Model(**cfg.model)
pipeline = Pipeline(model, dataset, **cfg.pipeline)

pipeline.run_test()
