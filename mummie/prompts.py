compostion_prompt = """
Give me a list of compositions if they are mentioned in the page if not return empty list. 
The compsition list should be of the type list[dict[str, float]]. Make sure they are realistic compositions containing real elements. 
If something is not parsed properly in the PDF, return the best guess of the composition.
"""

compostion_property_prompt = """
Give me a list of compositions and material properties if they are mentioned in the page if not return empty list. 
Make sure they are realistic compositions containing real elements. Check the consistency of units and make sure these are physical units.
If something is not parsed properly in the PDF, return the best guess of the composition.

Here is an example of the output:
example_output = [
    ({"SiO2": 50, "Al2O3": 50}, {"thermal_expansion": [{"value": 5, "unit": "W/mK", "experimental_conditions": "300K"}, {"value": 2.5, "unit": "W/mK", "experimental_conditions": "500K"}]}),
    ({"SiO2": 50, "Al2O3": 25, "MgO": 25}, {"viscosity": [{"value": 1e12, "unit": "Pa.s", "experimental_conditions": ""}]}),
    ({"Si": 50, "Na2O": 25, "K2O": 25}, {})
]
"""