"""Training-side data structures and utilities."""

from parity_posttrain.training.artifact import (
    extract_training_examples,
    load_training_examples,
)
from parity_posttrain.training.batch import (
    TrajectoryTrainingBatch,
    collate_training_examples,
)
from parity_posttrain.training.comparison import (
    TaskLogprobShift,
    TrainingComparisonRow,
    TrainingComparisonSummary,
    TrainingComparisonTask,
    training_comparison_to_dict,
)
from parity_posttrain.training.comparison_runner import (
    run_training_comparison,
    select_training_examples,
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
    PolicyNormalization,
    centered_reward_advantages,
    clipped_policy_loss,
)
from parity_posttrain.training.step import (
    TrainingStepResult,
    run_clipped_policy_step,
)

__all__ = [
    "ClippedPolicyLossResult",
    "IGNORE_INDEX",
    "PolicyNormalization",
    "TaskLogprobShift",
    "TrainingComparisonRow",
    "TrainingComparisonSummary",
    "TrainingComparisonTask",
    "TrainingStepResult",
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
    "run_clipped_policy_step",
    "run_training_comparison",
    "select_training_examples",
    "training_comparison_to_dict",
]
