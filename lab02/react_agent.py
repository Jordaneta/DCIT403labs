import os, re, sys, glob
from dotenv import load_dotenv
from litellm import completion
from tools import calculator, wiki_lookup
from litellm.exceptions import BadRequestError

load_dotenv()
MODEL = os.getenv("LLM_MODEL")

TOOLS = {"calculator": calculator, "wiki_lookup": wiki_lookup}

SYSTEM_PROMPT = """You are a helpful assistant for traders at Makola Market.
You can use these tools:
- calculator: evaluates arithmetic such as 4*185 + 3*95
- wiki_lookup: returns a Wikipedia summary for a page title
- finish: give the final answer to the user

IMPORTANT: Do NOT use function calling, tool calls or JSON. The tools are
used only through the plain text format below.

On every step reply in EXACTLY this format and nothing else:
Thought: <your reasoning>
Action: calculator OR wiki_lookup OR finish
Action Input: <input for the tool, or the final answer if the action is finish>

Example of one step:
Thought: I need the total cost first.
Action: calculator
Action Input: 4*185 + 3*95

Write only ONE step, then stop and wait for the Observation.
Never write an Observation yourself."""

ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\nAction Input:\s*(.*)", re.DOTALL)

def parse(text):
    m = ACTION_RE.search(text)
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def run_agent(question, max_steps=6):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    trace = [f"Question: {question}"]

    for step in range(max_steps):
        try:
            resp = completion(model=MODEL, messages=messages,
                              temperature=0, stop=["Observation:"])
            reply = resp.choices[0].message.content.strip()
        except BadRequestError:
            obs = ("Observation: Your last reply was rejected because you used "
                   "function calling. Reply in plain text using the "
                   "Thought / Action / Action Input format.")
            trace.append("[model reply rejected by API]")
            trace.append(obs)
            messages.append({"role": "user", "content": obs})
            continue

        trace.append(reply)
        action, action_input = parse(reply)
        if action == "finish":
            trace.append(f"FINAL ANSWER: {action_input}")
            break

        result = TOOLS[action](action_input)   # Step 5 will harden this line
        obs = f"Observation: {result}"
        trace.append(obs)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": obs})

    text = "\n".join(trace)
    print(text)
    os.makedirs("evidence", exist_ok=True)
    n = len(glob.glob("evidence/trace_*.txt")) + 1
    with open(f"evidence/trace_{n}.txt", "w") as f:
        f.write(text)

if __name__ == "__main__":
    run_agent(" ".join(sys.argv[1:]))