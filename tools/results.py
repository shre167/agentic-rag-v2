import json


def tool_result(
    name: str,
    input_value: str,
    result: str,
    passages: list[dict] | None = None,
    items: list[str] | None = None,
    latency_ms: float | None = None,
) -> str:
    """Serialize tool output for both the model and the request orchestrator."""
    data = {
        "tool": name,
        "input": input_value,
        "result": result,
        "passages": passages or [],
        "items": items or [],
    }
    if latency_ms is not None:
        data["latency_ms"] = latency_ms
    return json.dumps(data, ensure_ascii=False)
