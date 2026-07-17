"""Training-side data structures and utilities."""

from parity_posttrain.training.example import (
    IGNORE_INDEX,
    TrajectoryTrainingExample,
    build_generated_token_labels,
    build_generated_token_loss_mask,
    build_trajectory_training_example,
)

__all__ = [
    "IGNORE_INDEX",
    "TrajectoryTrainingExample",
    "build_generated_token_labels",
    "build_generated_token_loss_mask",
    "build_trajectory_training_example",
]
