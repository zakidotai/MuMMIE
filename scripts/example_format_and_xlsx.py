import json
import os

import pandas as pd

from mummie.schemas import Composition, CompositionPropertyList


if __name__ == "__main__":
    # 1) Minimal: parse the expected output format from a JSON string
    #    Structure: list of pairs [ [composition_dict, properties_dict], ... ]
    toy_data = [
    ({"SiO2": 50, "Al2O3": 50}, {"thermal_expansion": [{"value": 5, "unit": "W/mK", "experimental_conditions": "300K"}, {"value": 2.5, "unit": "W/mK", "experimental_conditions": "500K"}]}),
    ({"SiO2": 50, "Al2O3": 25, "MgO": 25}, {"viscosity": [{"value": 1e12, "unit": "Pa.s", "experimental_conditions": ""}]}),
    ({"Si": 50, "Na2O": 25, "K2O": 25}, {})
]
    toy_json = json.dumps(toy_data)
    parsed = CompositionPropertyList.model_validate_json(toy_json)
    item = parsed.root[1]
    print("Parsed toy item -> composition:", item.composition.root)
    print("Parsed toy item -> properties keys:", list(item.properties.root))
    print(parsed)