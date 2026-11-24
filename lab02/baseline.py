import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()
MODEL = os.getenv("LLM_MODEL")

BUDGET = ("I have GHS 1,200. I want 4 bags of rice at GHS 185 each and "
          "3 gallons of oil at GHS 95 each. How much will I have left?")
FACTS = ["What is the capital of the Bono East Region of Ghana?",
         "What is the population of the Techiman Municipality?"]

def ask(q):
    r = completion(model=MODEL, messages=[{"role": "user", "content": q}],
                   temperature=1)
    return r.choices[0].message.content.strip()

lines = ["# Baseline: model without tools", f"Model: {MODEL}", ""]
for i in range(3):
    lines += [f"## Budget question, run {i+1}", ask(BUDGET),
              "", "Correct (answer is GHS 175)? ____", ""]
for q in FACTS:
    lines += [f"## {q}", ask(q), "", "Correct? ____   Confidence of tone: ____", ""]

os.makedirs("evidence", exist_ok=True)
with open("evidence/baseline.md", "w") as f:
    f.write("\n".join(lines))
print("Saved evidence/baseline.md")