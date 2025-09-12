import os
import json
from mummie import configure_lm, MummieAgent
from mummie.prompts import compostion_property_prompt
import pandas as pd
import time

prompt_mapper_extraction = {
    'α*1E7, K-1： Thermal expansion coefficient temp.range, °C： Temperature range': 'CTE',
    'Glass transition temperature': 'Tg',
    'Density': 'rho',
    'T, °C： Temperature log(η, P)： η: Viscosity': 'T_η',
    "E: Young's modulus": "E",
    'Thermal expansion coefficient': 'CTE',
    'Refractive index': 'n',
    'H: Microhardness': 'H',
    'Crystallization temperature': 'Tx',
    'τ, %： Transmittance λ, nm： Wavelength': 'τ',
    'Liquidus temperature': 'Tl'
}


api_key = os.environ.get("CEREBRAS_API_KEY")
#api_key = os.environ.get("OPENAI_API_KEY")
model_name = 'qwen-3-235b-a22b-thinking-2507'
#model_name = 'llama-4-maverick-17b-128e-instruct'

#model_name = 'gpt-oss-120b'
#model_name = 'qwen-3-32b'

configure_lm(provider="cerebras", model=model_name, api_key=api_key, max_tokens=8192*8, temperature=0.0, timeout=200.0, top_p=0.90)

agent = MummieAgent(use_chain_of_thought=False)

minipdf_dir = '../data/pdf'
output_dir = '../data/output_normalized_prop'



pdf_files = [f for f in os.listdir(minipdf_dir) if f.endswith('.pdf')]

for pdf_filename in pdf_files:
    base_name = pdf_filename.replace('.pdf', '')
    md_filename = base_name + '.md'
    md_filepath = os.path.join('../data/marker_output', base_name, md_filename)
    output_json_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_json_dir, exist_ok=True)
    json_path = os.path.join(output_json_dir, f'{base_name}.json')
    # Skip if output already exists
    if os.path.exists(json_path):

        print(f"Output already exists for {pdf_filename}, skipping.")
        continue
    try:
        with open(md_filepath, 'r') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Markdown file not found for {pdf_filename}, skipping.")
        continue



    detailed_prompt = """
        Give me a list of compositions and material properties if they are mentioned in the text. If not, return an empty list.

        IMPORTANT: Only extract and include the following standardized material property keys (use these exact names):
        - CTE (thermal expansion coefficient)
        - Tg (glass transition temperature)
        - rho (density)
        - T_η (temperature for viscosity)
        - E (Young's modulus)
        - n (refractive index)
        - H (microhardness)
        - Tx (crystallization temperature)
        - τ (transmittance)
        - Tl (liquidus temperature)

        Your output MUST be a valid JSON object with a single key 'answer'.
        The value of 'answer' MUST be a list of lists, where each inner list contains:
        - a dictionary representing a composition (e.g. {"SiO2": 50, "Al2O3": 50})
        - a dictionary of material properties (e.g. {"CTE (thermal expansion coefficient)": [{"value": 5, "unit": "W/mK", "experimental_conditions": "300K"}]})

        Make sure they are realistic compositions containing real elements. Check the consistency of units and make sure these are physical units.
        If something is not parsed properly in the PDF, return your best guess in the required format.

        Do NOT return a plain dictionary. Do NOT return anything except the JSON object with the 'answer' key.
        Your response should ONLY be the JSON object in the format below. Do not include any explanation or extra text.

        Here is an example of the output:
        {"answer": [
            [{"SiO2": 50, "Al2O3": 50}, {"CTE (thermal expansion coefficient)": [{"value": 5, "unit": "W/mK", "experimental_conditions": "300K"}, {"value": 2.5, "unit": "W/mK", "experimental_conditions": "500K"}]}],
            [{"SiO2": 50, "Al2O3": 25, "MgO": 25}, {"rho (density)": [{"value": 2.5, "unit": "g/cm3", "experimental_conditions": ""}]}],
            [{"Si": 50, "Na2O": 25, "K2O": 25}, {}]
        ]}
        """
    snippet = text
    question = detailed_prompt

    prompt = f"text:\n{snippet}\n\nQuestion: {question}"
    import re
    def extract_json_answer(text):
        # Try to parse as JSON
        try:
            obj = json.loads(str(text))
            if isinstance(obj, dict) and 'answer' in obj:
                return {"answer": obj["answer"], "raw": text}
            elif isinstance(obj, dict):
                return {"answer": [obj], "raw": text}
            elif isinstance(obj, list):
                return {"answer": obj, "raw": text}
        except Exception:
            pass
        # Try to find any JSON object in the text
        import re
        match = re.search(r'(\{.*\})', str(text), re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(1))
                if isinstance(obj, dict) and 'answer' in obj:
                    return {"answer": obj["answer"], "raw": text}
                elif isinstance(obj, dict):
                    return {"answer": [obj], "raw": text}
            except Exception:
                pass
        # If it's a string, wrap it
        if isinstance(text, str):
            return {"answer": text, "raw": text}
        # If all fails, return null and error
        return {"answer": None, "error": str(text)}

    try:
        raw_answer = agent.ask(question=prompt)
        print(f"Response for {pdf_filename}: {raw_answer}")
        processed_answer = extract_json_answer(raw_answer)
        time.sleep(2)
    except Exception as e:
        print(f"Error processing {pdf_filename}: {e}")
        processed_answer = {"answer": None, "error": str(e)}

    json_path = os.path.join(output_json_dir, f'{base_name}.json')
    with open(json_path, "w") as f:
        json.dump(processed_answer, f, indent=2)


