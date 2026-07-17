from parity_posttrain.agent.runner import AgentRunner
from parity_posttrain.core.task import AgentTask
from parity_posttrain.evals.trajectory_evaluator import (
    evaluate_trajectory,
)
from parity_posttrain.rollout.hf_backend import GenerationResult


def make_generation(
    text: str,
    token_id: int,
) -> GenerationResult:
    return GenerationResult(
        model_name="fake-model",
        device="cpu",
        prompt_text="prompt",
        generated_text=text,
        prompt_token_ids=[1, 2],
        generated_token_ids=[token_id],
        generated_token_logprobs=[-0.1],
        latency_ms=10.0,
        tokens_per_second=100.0,
    )


class FakeBackend:
    def __init__(
        self,
        generations: list[GenerationResult],
    ) -> None:
        self.generations = generations.copy()
        self.calls: list[list[dict[str, str]]] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 32,
    ) -> GenerationResult:
        self.calls.append(
            [message.copy() for message in messages]
        )

        if not self.generations:
            raise RuntimeError("fake backend has no generation left")

        return self.generations.pop(0)


def build_calculator_task() -> AgentTask:
    return AgentTask(
        task_id="calculator_agent_test",
        prompt="Use the calculator to evaluate 17 * 6.",
        expected_answer="102",
        required_tools=["calculator"],
    )


def test_agent_runner_completes_tool_loop() -> None:
    backend = FakeBackend(
        [
            make_generation(
                '{"type":"tool_call","name":"calculator",'
                '"arguments":{"expression":"17 * 6"}}',
                10,
            ),
            make_generation(
                '{"type":"final","answer":"102"}',
                11,
            ),
        ]
    )
    runner = AgentRunner(
        backend,
        max_steps=3,
        max_new_tokens=64,
    )

    result = runner.run(build_calculator_task())

    assert result.status == "completed"
    assert result.error is None
    assert result.trajectory.token_ids == [10, 11]
    assert result.trajectory.token_logprobs == [-0.1, -0.1]
    assert result.trajectory.latency_ms == 20.0

    roles = [
        message.role
        for message in result.trajectory.messages
    ]
    assert roles == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.trajectory.messages[2].content == "102"
    assert result.trajectory.messages[-1].content == "102"

    evaluation = evaluate_trajectory(
        build_calculator_task(),
        result.trajectory,
    )
    assert evaluation.answer_correct is True
    assert evaluation.tool_coverage == 1.0
    assert evaluation.reward == 1.0

    assert len(backend.calls) == 2
    assert "TOOL_RESULT" in backend.calls[1][-1]["content"]
    assert "102" in backend.calls[1][-1]["content"]


def test_agent_runner_records_protocol_error() -> None:
    backend = FakeBackend(
        [
            make_generation(
                "The answer is probably 102.",
                20,
            )
        ]
    )
    runner = AgentRunner(backend)

    result = runner.run(build_calculator_task())

    assert result.status == "protocol_error"
    assert result.error is not None
    assert "valid JSON object" in result.error
    assert result.trajectory.messages[-1].role == "assistant"


def test_agent_runner_records_tool_error_and_recovers() -> None:
    backend = FakeBackend(
        [
            make_generation(
                '{"type":"tool_call","name":"calculator",'
                '"arguments":{"wrong_key":"17 * 6"}}',
                30,
            ),
            make_generation(
                '{"type":"final","answer":"0"}',
                31,
            ),
        ]
    )
    runner = AgentRunner(backend)

    result = runner.run(build_calculator_task())

    assert result.status == "completed"
    assert len(
        result.trajectory.metadata["tool_errors"]
    ) == 1
    assert "invalid arguments" in (
        result.trajectory.messages[2].content
    )
    assert "error" in backend.calls[1][-1]["content"]


def test_agent_runner_stops_at_max_steps() -> None:
    repeated_call = (
        '{"type":"tool_call","name":"calculator",'
        '"arguments":{"expression":"17 * 6"}}'
    )
    backend = FakeBackend(
        [
            make_generation(repeated_call, 40),
            make_generation(repeated_call, 41),
        ]
    )
    runner = AgentRunner(
        backend,
        max_steps=2,
    )

    result = runner.run(build_calculator_task())

    assert result.status == "max_steps"
    assert result.error == (
        "agent did not produce a final answer"
    )
    assert result.trajectory.messages[-1].role == "assistant"
    assert "maximum number of steps" in (
        result.trajectory.messages[-1].content
    )


def test_agent_runner_rejects_invalid_configuration() -> None:
    backend = FakeBackend([])

    try:
        AgentRunner(backend, max_steps=0)
    except ValueError as error:
        assert str(error) == "max_steps must be positive"
    else:
        raise AssertionError("expected ValueError")
