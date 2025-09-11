from pdf2image import convert_from_path
from mummie.core import PageQA
import dspy


dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

page_qa = dspy.Predict(PageQA)

pages = convert_from_path("../data/pdf/US20010014424A1.pdf", dpi=300)
total_responce = {}
for i, pil_image in enumerate(pages):
    question = "Give me a list of compositions if they are mentioned in the page if not return empty list.         The compsition list should be of the type list[dict[str, float]]. Make sure they are realistic compositions containing real elements.         If something is not parse properly in the PDF, return the best guess of the composition."
    raw_answer = page_qa(question=question, image=dspy.Image.from_PIL(pil_image))
    total_responce[i] = raw_answer
    print(f"Page {i+1}: {raw_answer}")