import matplotlib.pyplot as plt
import axelrod as axl
from openai import OpenAI
import inspect, os, argparse, json

from utils import *
from llm_strategy import LLM_strategy

def parse_opponent(opponent_str):
    if hasattr(axl, opponent_str):
        return getattr(axl, opponent_str)
    else:
        raise ValueError(f"Invalid opponent type: {opponent_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a tournament between the LLM strategy and the EvolvedANN strategy')
    parser.add_argument('--matches', type=int, default=3, help='Number of matches to run')
    parser.add_argument('--horizon', type=int, default=5, help='Number of turns per match')
    parser.add_argument('--verbose', type=bool, default=False, help='Print verbose output')
    parser.add_argument('--temp', type=float, default=0.5, help='Temperature for sampling')
    parser.add_argument('--models', nargs='+', default=["gpt-4o-mini", "gpt-4o"], 
                        help="List of GPT models to use")
    parser.add_argument('--opponent', type=str,
                        choices=['TitForTat', 'Adaptive', 'EvolvedANN'],
                        default="TitForTat",
                        help="Opponent (Axelrod strategy)")
    args = parser.parse_args()


    models=args.models
    opponent=parse_opponent(args.opponent)

    # Parse arguments
    matches = args.matches
    horizon = args.horizon
    verbose = args.verbose
    temp = args.temp

    # Build results directories
    run_name = build_run_name(horizon, matches, opponent.__name__)
    if not os.path.exists(f"results/{run_name}"):
        os.makedirs(f"results/{run_name}")


    # Create the players
    total_cumulative_scores = {model: [[0] * horizon, [0] * horizon] for model in models}

    for model in models:
        if not os.path.exists(f"results/{run_name}/{model}"):
            os.makedirs(f"results/{run_name}/{model}")

        for i in range(matches):
            player1 = LLM_strategy(horizon=horizon, verbose=verbose, model=model)
            player2 = opponent()

            match_name = build_match_name(model, i, player2.name)
            print(f"Match {i + 1} / {matches} for model {model}")

            # Simulate a match
            match = axl.Match([player1, player2], turns=horizon)
            results = match.play()

            # Update the cumulative scores
            scores = match.scores()
            cumulative_scores = [[sum(score[:i+1]) for i in range(len(score))] for score in zip(*scores)]
            total_cumulative_scores[model] = [[total_cumulative_scores[model][j][i] + cumulative_scores[j][i] for i in range(horizon)] for j in range(2)]

            # Print token usage
            print(f"Used {player1.token_usage} tokens for {model}")

            # Write llm log to file
            with open(f"results/{run_name}/{model}/match_{i}.json", "w") as f:
                json.dump(player1.log, f)

    # Make figs
    if not os.path.exists(f"results/{run_name}/figs"):
        os.makedirs(f"results/{run_name}/figs")
    fig_dir = f"results/{run_name}/figs"

    # Plot the average cumulative scores
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))

    for ax, model in zip(axs, models):
        ax.plot([score / matches for score in total_cumulative_scores[model][0]], label=f"{model}")
        ax.plot([score / matches for score in total_cumulative_scores[model][1]], label=player2.name)
        ax.legend()
        ax.set_xlabel('Round')
        ax.set_ylabel('Avg. Cumulative Score')
        ax.set_title(f"Repeated Prisoner's Dilemma, {player2.name} opponent. {matches} matches")

    plt.savefig(f'{fig_dir}/scores.png')

    # Plot all models stacked on top of each other
    plt.figure(figsize=(10, 6))
    for model in models:
        plt.plot([score / matches for score in total_cumulative_scores[model][0]], label=f"{model}")

    plt.legend()
    plt.xlabel('Turns')
    plt.ylabel('Cumulative Score')
    plt.title(f'Average Cumulative Scores, {matches} Matches, against {player2.name}')

    plt.savefig(f'{fig_dir}/stacked_comparison.png')

