import os
import json
from pypdf import PdfReader
from mummie import configure_lm, MummieAgent
import pandas as pd
from mummie.prompts import compostion_property_prompt


api_key = os.environ.get("CEREBRAS_API_KEY")

configure_lm(provider="cerebras", model="qwen-3-32b", api_key=api_key, max_tokens=8192, temperature=0.3, timeout=30.0, top_p=0.90)
agent = MummieAgent(use_chain_of_thought=True)

reader = PdfReader("../data/pdf/US20010014424A1.pdf")

total_responce = {}
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    snippet = text # keep prompt small
    question = compostion_property_prompt
    prompt = f"Page text:\n{snippet}\n\nQuestion: {question}"
    raw_answer = agent.ask(question=prompt)
    total_responce[i] = raw_answer
    print(f"Page {i+1}: {raw_answer}")

with open("total_responce.json", "w") as f:
    json.dump(total_responce, f)

df = pd.read_excel('../data/excel/US2009239122A1_table-30659.xlsx')
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