# README



## Project Overview

This project simulates a tournament between an LLM agent against various algorithms in an Iterated Prisoner's Dilemma. The LLM_strategy leverages OpenAI's language models to make decisions based on historical game data, as well as the code of the opponen't strategy.


### Prerequisites

1. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Set up your OpenAI API key as an environment variable:
   - Create a `.env` file in the root directory of your project.
   - Add the following line to it:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

### Usage

The main script is `main.py`. You can run it with various command-line arguments to customize the simulation:

```bash
python main.py --matches <number_of_matches> --horizon <turns_per_match> --verbose <True_or_False> --temp <temperature>
```

#### Command Line Arguments:

- **--matches**: Number of matches to run (default: 3)
- **--horizon**: Number of turns per match (default: 5)
- **--verbose**: Print verbose output (default: False)
- **--temp**: Temperature for sampling from the model (default: 0.5)

### Output

Results are saved in `/results`. Each match's log is saved as a JSON file containing the model's decision and reasoning at each round.

Additionally, two plots are generated and saved under `/results/<run_name>/figs`:
1. Average cumulative scores over rounds for each model.
2. Stacked comparison plot showing average cumulative scores across all models. (NOTE: stacked plot ignore's opponent's performance)


#### Example Command:

(I would just edit this line and copy+paste into the terminal)

```bash
python main.py --horizon 12 --matches 10 --verbose False --temp 0.5 --models gpt-4o-mini gpt-4o gpt-4 --opponent Adaptive
```

### File Descriptions

#### main.py

This script sets up and runs multiple matches between different strategies using specified configurations passed via command-line arguments.

#### llm_strategy.py

Defines the custom player class `LLM_strategy` which utilizes OpenAI's language models to decide its moves during matches based on historical game data and opponent behavior.
