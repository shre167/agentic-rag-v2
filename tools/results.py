import json


def tool_result(
    name: str,
    input_value: str,
    result: str,
    passages: list[dict] | None = None,
    items: list[str] | None = None,
) -> str:
    """Serialize tool output for both the model and the request orchestrator."""
    return json.dumps(
        {
            "tool": name,
            "input": input_value,
            "result": result,
            "passages": passages or [],
            "items": items or [],
        },
        ensure_ascii=False,
    )
