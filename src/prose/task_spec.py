"""离线 LLM 语义归纳所读取的任务描述。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VariableSpec:
    name: str
    unit: str = ""
    sampling_frequency: str = ""
    availability: str = "online"
    role: str = ""
    description: str = ""

    def to_prompt_block(self) -> str:
        return "\n".join(
            [
                f"- name: {self.name}",
                f"  unit: {self.unit or 'unknown'}",
                f"  sampling_frequency: {self.sampling_frequency or 'unknown'}",
                f"  availability: {self.availability}",
                f"  role: {self.role or 'unknown'}",
                f"  description: {self.description or 'not provided'}",
            ]
        )


@dataclass(frozen=True)
class TaskSpec:
    dataset: str
    target: str
    horizon: str
    online_boundary: str
    process_background: str
    variables: tuple[VariableSpec, ...]
    task_type: str = ""
    input_window_definition: str = ""

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> TaskSpec:
        variables = payload.get("variables")
        if not isinstance(variables, list) or not variables:
            raise ValueError("task spec must contain a non-empty variables list")
        fields = {name for name in VariableSpec.__dataclass_fields__}
        return cls(
            dataset=str(payload["dataset"]),
            target=str(payload["target"]),
            horizon=str(payload["horizon"]),
            online_boundary=str(payload["online_boundary"]),
            process_background=str(payload["process_background"]),
            variables=tuple(
                VariableSpec(**{key: str(value) for key, value in item.items() if key in fields})
                for item in variables
            ),
            task_type=str(payload.get("task_type", "")),
            input_window_definition=str(payload.get("input_window_definition", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["variables"] = [asdict(variable) for variable in self.variables]
        return payload


def load_task_spec(path: Path) -> TaskSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"task spec must be a JSON object: {path}")
    return TaskSpec.from_mapping(payload)
