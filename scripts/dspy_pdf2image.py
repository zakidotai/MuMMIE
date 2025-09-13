from pdf2image import convert_from_path
from mummie.core import PageQA
import dspy
import os
import base64
from io import BytesIO
import pandas as pd
from mummie.prompts import compostion_property_prompt

api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
dspy.configure(lm=dspy.LM("gemini/gemini-2.5-flash", api_key=api_key, max_tokens=
                          8192))

page_qa = dspy.Predict(PageQA)

pages = convert_from_path("../data/pdf/CN1308591A.pdf", dpi=300)
total_responce = {}
for i, pil_image in enumerate(pages[1:]):
    question = compostion_property_prompt
    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    raw_answer = page_qa(question=question, image=dspy.Image(url=data_url))
    total_responce[i] = raw_answer
    print(f"Page {i+1}: {raw_answer}")

df = pd.read_excel('../data/excel/CN1308591_table-24104.xlsx')
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