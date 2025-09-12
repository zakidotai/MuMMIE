import os
import json
from mummie import configure_lm, MummieAgent
import pandas as pd
import subprocess

api_key = os.environ.get("CEREBRAS_API_KEY")

configure_lm(provider="cerebras", model="qwen-3-235b-a22b-thinking-2507", api_key=api_key, max_tokens=8192*2, temperature=0.3, timeout=30.0, top_p=0.90)
agent = MummieAgent(use_chain_of_thought=False)

pdf_filename = "RU2016861C1.pdf"
base_name = pdf_filename.replace('.pdf', '')
md_filename = base_name + '.md'
md_filepath = os.path.join('../data/marker_output', base_name, md_filename)

with open(md_filepath, 'r') as f:
    text = f.read()
detailed_prompt="""Give me a list of compositions if they are mentioned in the text. If not, return an empty list.
The output should be a JSON object with a single key "answer".
The value of "answer" should be a list of dictionaries, where each dictionary represents a composition, like this: `{"answer": [{"element1": value1, "element2": value2}, ...]}`.
Make sure they are realistic compositions containing real elements. If something is not parsed properly in the PDF, return the best guess of the composition.
Your response should only be the JSON object."""
snippet = text # keep prompt small ##update the question
question = detailed_prompt
prompt = f"text:\n{snippet}\n\nQuestion: {question}"
raw_answer = agent.ask(question=prompt)
total_responce = {0: raw_answer}
print(f"Response: {raw_answer}")

with open("total_responce.json", "w") as f:
    json.dump(total_responce, f)

#df = pd.read_excel('../data/excel/US2009239122A1_table-30659.xlsx')
df = pd.read_excel('../data/excel/RU2016861_table-4306.xlsx')
column_dict = dict()
mapper = dict()
columns = df.columns.tolist()
idx = df[df[columns[0]] == 'Glass No'].index[-1]

idx0 = df[df[columns[0]] == 'Glass No'].index[0]
idx1 = df[df[columns[0]] == 'Glass No'].index[-1]

for k,v in zip(df.iloc[idx0], df.iloc[idx1]):
    k = k.replace("\n", ' ')
    if k in mapper:
        mapper[k].append(v)
    else:
        mapper[k] = [v]
dff = df.iloc[idx:,:]
dff.columns = df.iloc[idx]
dff = dff.iloc[1:,:]
# column_dict['CN1308591_table-24104'] = dff.columns.tolist()

print(dff.head())

