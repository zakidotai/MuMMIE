import os
import json
from pypdf import PdfReader
from mummie import configure_lm, MummieAgent
import pandas as pd

api_key = os.environ.get("CEREBRAS_API_KEY")

configure_lm(provider="cerebras", model="qwen-3-32b", api_key=api_key, max_tokens=8192, temperature=0.3, timeout=30.0, top_p=0.90)
agent = MummieAgent(use_chain_of_thought=True)

reader = PdfReader("../data/pdf/US1304623.pdf")

total_responce = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    snippet = text # keep prompt small
    question = "Give me a list of compositions if they are mentioned in the page if not return empty list. \
        The compsition list should be of the type list[dict[str, float]]. Make sure they are realistic compositions containing real elements. \
        If something is not parse properly in the PDF, return the best guess of the composition."
    prompt = f"Page text:\n{snippet}\n\nQuestion: {question}"
    raw_answer = agent.ask(question=prompt)
    total_responce.append(raw_answer)
    print(f"Page {i+1}: {raw_answer}")

with open("total_responce.json", "w") as f:
    json.dump(total_responce, f)

df = pd.read_excel('../data/excel/US1304623_table-24764.xlsx')
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
column_dict['US1304623_table-24764'] = dff.columns.tolist()

print(dff.head())