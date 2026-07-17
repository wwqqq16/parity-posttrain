from parity_posttrain.data.task_factory import build_sample_tasks


def test_sample_task_count() -> None:
    tasks = build_sample_tasks()

    assert len(tasks) == 24


def test_sample_task_ids_are_unique() -> None:
    tasks = build_sample_tasks()
    task_ids = [task.task_id for task in tasks]

    assert len(task_ids) == len(set(task_ids))


def test_sample_tasks_are_deterministic() -> None:
    first = [task.to_dict() for task in build_sample_tasks()]
    second = [task.to_dict() for task in build_sample_tasks()]

    assert first == second


def test_sample_tasks_cover_expected_categories() -> None:
    tasks = build_sample_tasks()
    categories = {task.metadata["category"] for task in tasks}

    assert categories == {
        "calculator",
        "catalog",
        "currency",
        "shopping",
        "basket",
    }


def test_sample_tasks_include_multi_tool_tasks() -> None:
    tasks = build_sample_tasks()

    assert any(len(task.required_tools) == 2 for task in tasks)
    assert any(len(task.required_tools) == 3 for task in tasks)


def test_all_sample_tasks_are_valid() -> None:
    for task in build_sample_tasks():
        task.validate()
