import os

from mummie import CerebrasAgent

api_key = os.environ.get("CEREBRAS_API_KEY")
agent = CerebrasAgent(api_key, model="qwen-3-32b", max_tokens=2048, temperature=0.7, timeout=60.0, top_p=0.95)
print(agent.ask("Give a one-line fun fact about AI."))
