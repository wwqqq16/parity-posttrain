"""Training-side data structures and utilities."""

from parity_posttrain.training.artifact import (
    extract_training_examples,
    load_training_examples,
)
from parity_posttrain.training.batch import (
    TrajectoryTrainingBatch,
    collate_training_examples,
)
from parity_posttrain.training.example import (
    IGNORE_INDEX,
    TrajectoryTrainingExample,
    build_generated_token_labels,
    build_generated_token_loss_mask,
    build_trajectory_training_example,
)
from parity_posttrain.training.logprobs import (
    gather_masked_causal_logprobs,
    rescore_training_batch,
)
from parity_posttrain.training.objective import (
    ClippedPolicyLossResult,
    centered_reward_advantages,
    clipped_policy_loss,
)

__all__ = [
    "ClippedPolicyLossResult",
    "IGNORE_INDEX",
    "TrajectoryTrainingBatch",
    "TrajectoryTrainingExample",
    "build_generated_token_labels",
    "build_generated_token_loss_mask",
    "build_trajectory_training_example",
    "centered_reward_advantages",
    "clipped_policy_loss",
    "collate_training_examples",
    "extract_training_examples",
    "gather_masked_causal_logprobs",
    "load_training_examples",
    "rescore_training_batch",
]
