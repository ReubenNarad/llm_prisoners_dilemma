from datetime import datetime

def build_run_name(horizon, n_matches, opponent_name):

    time = datetime.now().strftime("%m-%d_%H-%M-%S")
    return f"horizon_{horizon}_matches_{n_matches}_{opponent_name}"

def build_match_name(model, i, opponent_name):
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{model}_vs_{opponent_name}_match_{i}_{time}"
