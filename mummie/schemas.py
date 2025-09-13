from __future__ import annotations

from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field, RootModel, field_validator, model_validator


class Measurement(BaseModel):
    """A single measured material property value with unit and optional conditions."""

    value: float = Field(..., description="Numeric value of the measurement")
    unit: str = Field(..., description="Unit symbol or name, e.g. 'Pa.s', 'W/mK'")
    experimental_conditions: str = Field(
        default="", description="Free-text experimental conditions, e.g. temperature or method"
    )


class Properties(RootModel[Dict[str, List[Measurement]]]):
    """Mapping from property name to a list of measurements.

    Example: {"thermal_expansion": [Measurement(...), ...]}
    """

    root: Dict[str, List[Measurement]]


class Composition(RootModel[Dict[str, float]]):
    """Composition as a mapping of component name to quantity (typically wt% or mol%)."""

    root: Dict[str, float]

    @field_validator("root")
    @classmethod
    def _validate_non_negative(cls, v: Dict[str, float]) -> Dict[str, float]:
        for component, amount in v.items():
            if not isinstance(amount, (int, float)):
                raise TypeError(f"Amount for component '{component}' must be numeric, got {type(amount)}")
            if amount < 0:
                raise ValueError(f"Amount for component '{component}' must be non-negative, got {amount}")
        return v

    def normalized(self) -> "Composition":
        """Return a new Composition normalized to sum to 100 if the sum is positive."""
        total = float(sum(self.root.values()))
        if total <= 0:
            return self
        return Composition(root={k: (v / total) * 100.0 for k, v in self.root.items()})


class CompositionPropertyPair(BaseModel):
    """One item consisting of a composition and its associated measured properties."""

    composition: Composition
    properties: Properties


class CompositionPropertyList(RootModel[List[CompositionPropertyPair]]):
    """List of (composition, properties) items.

    Accepts either a list of dict objects with keys 'composition' and 'properties',
    or a list of 2-tuples (composition, properties) as shown in the prompt example.
    """

    root: List[CompositionPropertyPair]

    @model_validator(mode="before")
    @classmethod
    def _coerce_tuples(cls, data: Any) -> Any:
        # Allow inputs like: [(composition_dict, properties_dict), ...]
        if isinstance(data, list):
            coerced: List[Union[CompositionPropertyPair, Dict[str, Any]]] = []
            for item in data:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    comp, props = item
                    coerced.append({"composition": comp, "properties": props})
                else:
                    coerced.append(item)
            return coerced
        return data