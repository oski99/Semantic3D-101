import os
import argparse
import logging
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d


def configure_logging():
    """Configure logging format and level"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
    )

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='3D Point Cloud Semantic Segmentation Training')
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='Semantic3D/processed',
        help='Path to the dataset directory'
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=None,
        help='Path to checkpoint file to load (for resuming training)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/randlanet_semantic3d.yml',
        help='Path to config file (default: randlanet_semantic3d.yml)'
    )
    parser.add_argument(
        '--framework',
        type=str,
        default='torch',
        choices=['torch', 'tf'],
        help='Deep learning framework to use (default: torch)'
    )
    return parser.parse_args()

def main():
    # Configure logging
    configure_logging()

    # Parse arguments
    args = parse_arguments()

    try:
        # Load configuration
        cfg = _ml3d.utils.Config.load_from_file(args.config)

        # Update paths from command line arguments
        cfg.dataset['dataset_path'] = args.dataset_path

        # Handle checkpoint paths
        if args.ckpt_path:
            if not os.path.exists(args.ckpt_path):
                raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt_path}")
            cfg.model['ckpt_path'] = args.ckpt_path
            logging.info(f"Will resume training from checkpoint: {args.ckpt_path}")

        # Verify paths exist
        if not os.path.exists(args.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")

        # Initialize components
        Pipeline = _ml3d.utils.get_module("pipeline", cfg.pipeline.name, args.framework)
        Model = _ml3d.utils.get_module("model", cfg.model.name, args.framework)
        Dataset = _ml3d.utils.get_module("dataset", cfg.dataset.name)

        # Create instances
        dataset = Dataset(cfg.dataset.pop('dataset_path', None), **cfg.dataset)
        model = Model(**cfg.model)
        pipeline = Pipeline(model, dataset, **cfg.pipeline)

        pipeline.run_train()

        logging.info("Training completed successfully")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()