# Lab 2: Architectures: A ReAct Agent From Scratch: Concept Check

Answer in your own words before the Wednesday session (independent work). Your answers feed Section 2 of your lab report (Concept and classical root).

## Q1. Classify a thermostat, a chess engine and a ReAct agent as reactive, deliberative or hybrid, with one reason each.
- A thermostat is 'reactive'. It maps the current percept (the temperature) straight to an action with a fixed condition-action rule. It has no model of the world and makes no plan.
- A chess engine is 'deliberative'. It keeps a model of the game and searches ahead through possible move sequences to pick the one that best serves its goal of winning.
- A ReAct agent is 'hybrid'. The Thought step is deliberative, because the model works out what it still needs to know. Each tool call and the reaction to its Observation is reactive, because the agent responds to a new percept straight away. The whole loop mixes both


## Q2. In the trace above, which lines are percepts, and which are actions?
The percepts are the 'Observation:' lines (and the user's original question), because they are what the agent receives from the outside world. The actions are the 'Action:' and 'Action Input:' lines, because they are what the agent does to the world (running a tool, or giving the final answer with finish). The 'Thought:' lines are neither. They are internal deliberation where the model updates its plan.


## Q3. Why do we pass `stop=["Observation:"]` when calling the model?

The model only writes text. After an Action it would happily keep going and invent the Observation itself, which would be a made-up tool result instead of a real one. Stopping generation at that string halts the model right after the Action Input. My code then runs the real tool and adds the real Observation, so the percept is true. It also saves tokens.


## Q4. Map the trace above to the BDI terms belief, desire and intention.

- "Belief:" the conversation so far, including every observation (for example, the cost is 1025 after the first calculator call).
- "Desire:" the user's goal, to find out how much money is left.
- "Intention:" the plan the model states in its latest Thought, such as "now subtract the cost from the budget". It is committed to that next action.

