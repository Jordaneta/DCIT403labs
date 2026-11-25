import os, re, sys, glob, time
from dotenv import load_dotenv
from litellm import completion
from litellm.exceptions import BadRequestError, RateLimitError
from tools import calculator, wiki_lookup

load_dotenv()
MODEL = os.getenv("LLM_MODEL")

TOOLS = {"calculator": calculator, "wiki_lookup": wiki_lookup}

SYSTEM_PROMPT = """You are a helpful assistant for traders at Makola Market.
You can use these tools:
- calculator: evaluates arithmetic such as 4*185 + 3*95
- wiki_lookup: returns a Wikipedia summary for a page title
- finish: give the final answer to the user

Rules:
- For ANY arithmetic, even simple multiplication, write "Action: calculator".
- For ANY factual question about a place, person or thing, write
  "Action: wiki_lookup".
- Never answer from memory. Every fact or number in your final answer must
  come from an Observation.
- If the question has two parts, handle one part per step, and use finish
  only after both parts have an Observation.

IMPORTANT: Never use function calling, tool calls or JSON. Your whole reply
must be plain text in the format below. The words calculator and wiki_lookup
are written after "Action:", they are not functions you can call.

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


def call_model(messages):
    """Call the model; if the free tier rate limit is hit, wait and retry."""
    for attempt in range(4):
        try:
            return completion(model=MODEL, messages=messages,
                              temperature=0, stop=["Observation:"])
        except RateLimitError:
            wait = 15 * (attempt + 1)
            print(f"[rate limit hit, waiting {wait}s]")
            time.sleep(wait)
    return completion(model=MODEL, messages=messages,
                      temperature=0, stop=["Observation:"])


def run_agent(question, max_steps=6):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    trace = [f"Question: {question}"]
    finished = False
    steps = 0
    rejections = 0
    stop_reason = None
    seen = set()

    while steps < max_steps:
        try:
            resp = call_model(messages)
        except BadRequestError:
            rejections += 1
            trace.append("[model reply rejected by API: used function calling]")
            if rejections >= 3:
                stop_reason = "STOPPED: model kept using function calling (3 rejections)."
                break
            # keep the conversation in order: add the note to the last message
            messages[-1]["content"] += (
                "\n\n(Note: your previous attempt was rejected because it used "
                "function calling. Reply in plain text only, using the "
                "Thought / Action / Action Input format, and continue from "
                "where you were.)")
            trace.append("[correction added to the conversation]")
            continue

        steps += 1
        reply = resp.choices[0].message.content.strip()
        trace.append(reply)
        action, action_input = parse(reply)

        # Case 1: reply cannot be parsed
        if action is None:
            obs = ("Observation: Could not parse your reply. Reply with exactly "
                   "one step in this format:\nThought: ...\n"
                   "Action: calculator OR wiki_lookup OR finish\n"
                   "Action Input: ...")
        # Finished
        elif action == "finish":
            trace.append(f"FINAL ANSWER: {action_input}")
            finished = True
            break
        # Case 2: unknown action name
        elif action not in TOOLS:
            obs = (f"Observation: Unknown action '{action}'. Valid actions are: "
                   "calculator, wiki_lookup, finish.")
        # Case 3: the same action repeated (loop detection)
        elif (action, action_input) in seen:
            obs = ("Observation: You already ran this exact action and have the "
                   "result above. Do the NEXT part of the question with a "
                   "different action, or use finish if you have everything.")
        # Normal tool call
        else:
            seen.add((action, action_input))
            try:
                obs = f"Observation: {TOOLS[action](action_input)}"
            except Exception as e:
                obs = f"Observation: Tool failed: {e}"

        trace.append(obs)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": obs})

    # Case 4: ran out of steps (or too many rejections)
    if not finished:
        trace.append(stop_reason or
                     f"STOPPED: reached max_steps ({max_steps}) without finishing.")

    text = "\n".join(trace)
    print(text)
    os.makedirs("evidence", exist_ok=True)
    n = len(glob.glob("evidence/trace_*.txt")) + 1
    with open(f"evidence/trace_{n}.txt", "w") as f:
        f.write(text)


if __name__ == "__main__":
    run_agent(" ".join(sys.argv[1:]))