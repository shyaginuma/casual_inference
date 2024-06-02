from dataclasses import dataclass


@dataclass
class CustomMetric:
    name: str
    denominator: str
    numerator: str
