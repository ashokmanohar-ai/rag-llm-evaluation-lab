"""In-process metric recorder used by adapters and tests."""

from dataclasses import dataclass, field


@dataclass
class MetricsRecorder:
    observations: dict[str, list[float]] = field(default_factory=dict)

    def record(self, name: str, value: float) -> None:
        self.observations.setdefault(name, []).append(value)
