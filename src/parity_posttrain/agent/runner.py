"""Multi-turn model and tool execution loop."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from parity_posttrain.agent.protocol import (
    FinalAnswerAction,
    ToolCallAction,
    parse_agent_action,
)
from parity_posttrain.agent.tool_registry import (
    execute_tool_call,
    get_tool_schemas,
)
from parity_posttrain.core.task import AgentTask
from parity_posttrain.core.trajectory import Message, Trajectory
from parity_posttrain.rollout.hf_backend import GenerationResult


class RolloutBackend(Protocol):
    """Interface required by the agent runner."""

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 32,
    ) -> GenerationResult:
        """Generate one assistant response."""


type AgentRunStatus = Literal[
    "completed",
    "protocol_error",
    "max_steps",
]


@dataclass(slots=True)
class AgentRunResult:
    """Result of one multi-turn model and tool execution."""

    status: AgentRunStatus
    trajectory: Trajectory
    generations: list[GenerationResult]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def build_agent_system_prompt() -> str:
    """Build the strict JSON protocol prompt for the model."""

    schemas = json.dumps(
        get_tool_schemas(),
        indent=2,
        sort_keys=True,
    )

    return (
        "You are a tool-using agent operating in a strict action loop.\n"
        "At every turn, output exactly one JSON object and no other text.\n\n"
        "To call a tool, use:\n"
        '{"type":"tool_call","name":"TOOL_NAME",'
        '"arguments":{"ARGUMENT":"VALUE"}}\n\n'
        "To finish, use:\n"
        '{"type":"final","answer":"FINAL_ANSWER"}\n\n'
        "Rules:\n"
        "1. Use the required tool instead of solving the task mentally.\n"
        "2. Use only tools listed below.\n"
        "3. After receiving TOOL_RESULT, either call another tool or "
        "return a final answer.\n"
        "4. The final answer must use information from tool results.\n"
        "5. Do not use Markdown fences or explanations.\n\n"
        "Available tools:\n"
        f"{schemas}\n\n"
        "Example:\n"
        "User asks: Calculate 2 + 3.\n"
        "Assistant responds:\n"
        '{"type":"tool_call","name":"calculator",'
        '"arguments":{"expression":"2 + 3"}}\n'
        "After TOOL_RESULT 5, assistant responds:\n"
        '{"type":"final","answer":"5"}'
    )


def format_tool_feedback(
    *,
    tool_name: str,
    tool_content: str,
) -> str:
    """Format a tool result as the next model input."""

    payload = {
        "name": tool_name,
        "content": tool_content,
    }

    return (
        "TOOL_RESULT\n"
        f"{json.dumps(payload, sort_keys=True)}\n"
        "Respond with exactly one JSON action."
    )


class AgentRunner:
    """Run a model through a bounded multi-turn tool-use loop."""

    def __init__(
        self,
        backend: RolloutBackend,
        *,
        max_steps: int = 4,
        max_new_tokens: int = 128,
    ) -> None:
        """Configure the rollout loop."""

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        self.backend = backend
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens

    def _build_result(
        self,
        *,
        task: AgentTask,
        status: AgentRunStatus,
        messages: list[Message],
        generations: list[GenerationResult],
        tool_errors: list[str],
        error: str | None = None,
    ) -> AgentRunResult:
        """Build and validate the final trajectory."""

        token_ids = [
            token_id
            for generation in generations
            for token_id in generation.generated_token_ids
        ]
        token_logprobs = [
            logprob
            for generation in generations
            for logprob in generation.generated_token_logprobs
        ]
        total_latency_ms = round(
            sum(
                generation.latency_ms
                for generation in generations
            ),
            3,
        )

        metadata: dict[str, Any] = {
            "status": status,
            "generation_count": len(generations),
            "tool_errors": tool_errors,
        }

        if generations:
            metadata["model_name"] = generations[0].model_name
            metadata["device"] = generations[0].device

        trajectory = Trajectory(
            task_id=task.task_id,
            messages=messages,
            latency_ms=total_latency_ms,
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            metadata=metadata,
        )
        trajectory.validate()

        return AgentRunResult(
            status=status,
            trajectory=trajectory,
            generations=generations,
            error=error,
        )

    def run(self, task: AgentTask) -> AgentRunResult:
        """Run one task until a final answer or stopping condition."""

        task.validate()

        model_messages = [
            {
                "role": "system",
                "content": build_agent_system_prompt(),
            },
            {
                "role": "user",
                "content": task.prompt,
            },
        ]
        trajectory_messages = [
            Message(
                role="user",
                content=task.prompt,
            )
        ]
        generations: list[GenerationResult] = []
        tool_errors: list[str] = []

        for step in range(1, self.max_steps + 1):
            generation = self.backend.generate(
                model_messages,
                max_new_tokens=self.max_new_tokens,
            )
            generations.append(generation)

            try:
                action = parse_agent_action(
                    generation.generated_text
                )
            except ValueError as parse_error:
                trajectory_messages.append(
                    Message(
                        role="assistant",
                        content=generation.generated_text,
                    )
                )

                return self._build_result(
                    task=task,
                    status="protocol_error",
                    messages=trajectory_messages,
                    generations=generations,
                    tool_errors=tool_errors,
                    error=(
                        f"invalid agent action at step {step}: "
                        f"{parse_error}"
                    ),
                )

            if isinstance(action, FinalAnswerAction):
                trajectory_messages.append(
                    Message(
                        role="assistant",
                        content=action.answer,
                    )
                )

                return self._build_result(
                    task=task,
                    status="completed",
                    messages=trajectory_messages,
                    generations=generations,
                    tool_errors=tool_errors,
                )

            if isinstance(action, ToolCallAction):
                tool_call = {
                    "name": action.name,
                    "arguments": action.arguments.copy(),
                }
                trajectory_messages.append(
                    Message(
                        role="assistant",
                        content=generation.generated_text,
                        tool_call=tool_call,
                    )
                )
                model_messages.append(
                    {
                        "role": "assistant",
                        "content": generation.generated_text,
                    }
                )

                try:
                    execution = execute_tool_call(tool_call)
                    tool_content = (
                        execution.to_message_content()
                    )
                    tool_name = execution.name
                except ValueError as tool_error:
                    error_message = str(tool_error)
                    tool_errors.append(error_message)
                    tool_name = action.name
                    tool_content = json.dumps(
                        {"error": error_message},
                        sort_keys=True,
                    )

                trajectory_messages.append(
                    Message(
                        role="tool",
                        name=tool_name,
                        content=tool_content,
                    )
                )
                model_messages.append(
                    {
                        "role": "user",
                        "content": format_tool_feedback(
                            tool_name=tool_name,
                            tool_content=tool_content,
                        ),
                    }
                )

        trajectory_messages.append(
            Message(
                role="assistant",
                content=(
                    "Agent stopped after reaching the maximum "
                    "number of steps."
                ),
            )
        )

        return self._build_result(
            task=task,
            status="max_steps",
            messages=trajectory_messages,
            generations=generations,
            tool_errors=tool_errors,
            error="agent did not produce a final answer",
        )
