"""MLflow setup and utilities."""

import logging
from contextlib import contextmanager

import mlflow
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)


def setup_mlflow(config):
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)

    try:
        experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
        if experiment is None:
            mlflow.create_experiment(config.mlflow.experiment_name)
    except MlflowException as exc:
        logger.error(
            "Failed to initialize MLflow experiment '%s': %s",
            config.mlflow.experiment_name,
            exc,
        )
        raise

    mlflow.set_experiment(config.mlflow.experiment_name)


def log_config(config):
    """Log configuration to MLflow."""
    from omegaconf import OmegaConf

    config_dict = OmegaConf.to_container(config, resolve=True)
    _log_dict_params(config_dict)


def _log_dict_params(d, prefix=""):
    """Recursively log dictionary parameters."""
    for key, value in d.items():
        if isinstance(value, dict):
            _log_dict_params(value, f"{prefix}{key}.")
        else:
            try:
                mlflow.log_param(f"{prefix}{key}", value)
            except (MlflowException, TypeError) as exc:
                logger.warning("Skipping MLflow param %s%s: %s", prefix, key, exc)


@contextmanager
def start_run(run_name, tags=None):
    """Context manager for MLflow run."""
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run
