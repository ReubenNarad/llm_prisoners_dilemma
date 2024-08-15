import axelrod as axl
import inspect, os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") # Or enter manually
client = OpenAI(api_key=api_key)

# ANN explanation to add to the EvolvedANN strategy
ann_explanation = '''
Artificial Neural Network based strategy.

    A single layer neural network based strategy, with the following
    features:
    * Opponent's first move is C
    * Opponent's first move is D
    * Opponent's second move is C
    * Opponent's second move is D
    * Player's previous move is C
    * Player's previous move is D
    * Player's second previous move is C
    * Player's second previous move is D
    * Opponent's previous move is C
    * Opponent's previous move is D
    * Opponent's second previous move is C
    * Opponent's second previous move is D
    * Total opponent cooperations
    * Total opponent defections
    * Total player cooperations
    * Total player defections
    * Round number'''

class LLM_strategy(axl.Player):
    def __init__(self, horizon=3, verbose=True, model='gp-4o-mini', temp=0.5):
        super().__init__()
        self.token_usage = 0
        self.horizon = horizon
        self.verbose = verbose
        self.model = self.name = model
        self.temp = temp
        self.log = []

    def reset(self):
        self.token_usage = 0
        self.log = []

    def strategy(self, opponent):
        # Get match history
        opp_history = opponent.history
        my_history = self.history
        round = len(my_history)

        history_str = ""
        for i, (me, opp) in enumerate(zip(my_history, opp_history)):
            # Calculate reward
            if me == axl.Action.C:
                if opp == axl.Action.C:
                    reward = 3
                else:
                    reward = 0
            else:
                if opp == axl.Action.C:
                    reward = 5
                else:
                    reward = 3
            history_str += f"\nRound {i + 1} / {self.horizon}, reward {reward}\nMy move: {me}, Opponent's move: {opp}"
        
        if self.verbose:
            print(f"Round {round} / {self.horizon}: {[(my_history[i], opp_history[i]) for i in range(round)]}", end="\r")
        # Get opponent's source code as string
        source_code = inspect.getsource(opponent.strategy)

        if type(opponent) == axl.EvolvedANN:
            source_code = ann_explanation + source_code

        # Add messages (THIS IS THE PROMPT)
        messages = [{'role': 'user', 'content': f'''You are playing a repeated Prisoner's dilemma, with the choice of either Cooperating (C) or Defecting (D).
The payoff matrix is as follows: If both players defect, they both recieve 1. If both players cooperate, they both recieve 3. If one player defects and the other cooperates, the defector recieves 5, and the cooperator recieves 0.
Your opponent is a python agent playing according to this code:
```python{source_code}```
This is the game's history so far:
{history_str or "Round 0 / " + str(self.horizon)}
Your goal is to maximize your LONG TERM REWARD. Consider the value of exploration.
Do you (C) or defect (D) this round? Respond STRICTLY with 'C\n' or 'D\n', followed by your (brief) reasoning.'''
        }]

        # Make the API call
        for attempt in range(3):
            response = client.chat.completions.create(messages=messages, model=self.model)
            content = response.choices[0].message.content.strip().split('\n')
            self.token_usage += response.usage.total_tokens

            if self.verbose:
                print(content)

            # If neither 'C' nor 'D' is in the response, try again
            if 'C' not in content[0] and 'D' not in content[0] or len(content) < 2:
                if self.verbose:
                    print("Model Refusal / Invalid Response, trying again...")
                continue

            # Parse the response
            decision = axl.Action.C if 'C' in content[0] else axl.Action.D
            reasoning = "".join(content[1:])

            self.log.append({
                'round': round,
                'decision': str(decision),
                'reasoning': reasoning
            })

            return decision

        print("Giving up after 3 attempts, returning C")
        return axl.Action.C