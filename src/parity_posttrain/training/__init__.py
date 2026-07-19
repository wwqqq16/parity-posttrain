"""Training-side data structures and utilities."""

from parity_posttrain.training.artifact import (
    extract_training_examples,
    load_training_examples,
)
from parity_posttrain.training.batch import (
    TrajectoryTrainingBatch,
    collate_training_examples,
)
from parity_posttrain.training.closed_loop import (
    ClosedLoopAggregate,
    ClosedLoopSummary,
    ClosedLoopTaskComparison,
    ClosedLoopTaskSnapshot,
    build_closed_loop_aggregate,
    closed_loop_summary_to_dict,
)
from parity_posttrain.training.closed_loop_artifact import (
    extract_closed_loop_snapshots,
    load_closed_loop_snapshots,
)
from parity_posttrain.training.closed_loop_rerollout import (
    rerollout_agent_tasks,
)
from parity_posttrain.training.closed_loop_runner import (
    run_closed_loop_experiment,
)
from parity_posttrain.training.comparison import (
    TaskLogprobShift,
    TrainingComparisonRow,
    TrainingComparisonStep,
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
from parity_posttrain.training.loop import (
    TrainingLoopResult,
    run_clipped_policy_training,
)
from parity_posttrain.training.objective import (
    ClippedPolicyLossResult,
    PolicyNormalization,
    centered_reward_advantages,
    clipped_policy_loss,
)
from parity_posttrain.training.parameters import (
    TrainableParameterSelection,
    prepare_trainable_parameters,
)
from parity_posttrain.training.step import (
    TrainingStepResult,
    run_clipped_policy_step,
)

__all__ = [
    "ClippedPolicyLossResult",
    "ClosedLoopAggregate",
    "ClosedLoopSummary",
    "ClosedLoopTaskComparison",
    "ClosedLoopTaskSnapshot",
    "IGNORE_INDEX",
    "PolicyNormalization",
    "TaskLogprobShift",
    "TrainingComparisonRow",
    "TrainingComparisonStep",
    "TrainingComparisonSummary",
    "TrainingComparisonTask",
    "TrainableParameterSelection",
    "TrainingLoopResult",
    "TrainingStepResult",
    "TrajectoryTrainingBatch",
    "TrajectoryTrainingExample",
    "build_generated_token_labels",
    "build_generated_token_loss_mask",
    "build_closed_loop_aggregate",
    "build_trajectory_training_example",
    "closed_loop_summary_to_dict",
    "centered_reward_advantages",
    "clipped_policy_loss",
    "collate_training_examples",
    "extract_closed_loop_snapshots",
    "extract_training_examples",
    "gather_masked_causal_logprobs",
    "load_closed_loop_snapshots",
    "load_training_examples",
    "prepare_trainable_parameters",
    "rerollout_agent_tasks",
    "rescore_training_batch",
    "run_clipped_policy_step",
    "run_closed_loop_experiment",
    "run_clipped_policy_training",
    "run_training_comparison",
    "select_training_examples",
    "training_comparison_to_dict",
]
