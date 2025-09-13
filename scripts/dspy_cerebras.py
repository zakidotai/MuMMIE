import os

from mummie import configure_lm, MummieAgent

api_key = os.environ.get("CEREBRAS_API_KEY")
configure_lm(provider="cerebras", model="qwen-3-32b", api_key=api_key, max_tokens=2048, temperature=0.7, timeout=60.0, top_p=0.95)

agent = MummieAgent(use_chain_of_thought=True)
print(agent.ask("Give a one-line fun fact about AI."))
