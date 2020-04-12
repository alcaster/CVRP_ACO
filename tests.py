import pickle
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

from aco_tsp_reworked import Config, run_aco, Solution

N_AVERAGE = 2


def main():
    baseline_config, two_opt_config, candidate_list_config = Config(), Config(), Config()

    baseline_config.USE_2_OPT_STRATEGY = False
    two_opt_config.USE_2_OPT_STRATEGY = True
    candidate_list_config.USE_2_OPT_STRATEGY = False

    baseline_config.USE_CANDIDATE_LIST_STRATEGY = False
    two_opt_config.USE_2_OPT_STRATEGY = False
    candidate_list_config.USE_2_OPT_STRATEGY = True

    configs: Dict[str, Config] = {
        "baseline": baseline_config,
        "two_opt": two_opt_config,
        "candidate_list": candidate_list_config,
    }

    n_ants = [20, 40, 100]
    alphas = [1, 2, 4, 5]
    rhos = [0.05, 0.1, 0.25, 0.3]

    timestamp = int(time.time())
    log_dir = Path(__file__).resolve().parent / f"results_{timestamp}"
    log_dir.mkdir()

    results: Dict[Tuple, Tuple[np.ndarray, List[np.ndarray]]] = {}
    for name, config in configs.items():
        print(f"Testing config: {name}")
        for ants in n_ants:
            print(f"Ants: {ants}")
            config.NUM_ANTS = ants
            for alpha in alphas:
                print(f"Alpha: {alpha}")
                config.ALPHA = alpha
                for rho in rhos:
                    experiment_name = f"{name.upper()}: ants: {ants}, alpha: {alpha}, rho: {rho}"
                    print(experiment_name)
                    config.RHO = rho

                    curr_results = []
                    for _ in range(N_AVERAGE):
                        result = run_aco(config, False)
                        curr_results.append(result)
                    best, history = average_n_results(curr_results)

                    with open(log_dir / "log.txt", 'a') as f:
                        f.write(f"{experiment_name} = {best}\n")
                    results[name, ants, alpha, rho] = (best, history)

    with open(log_dir / "save.pkl", 'wb') as f:
        pickle.dump(results, f)


def average_n_results(results: List[Tuple[Solution, List[Solution]]]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Average n results. Drop paths.
    :param results:
    :return: Tuple best score and history of scores
    """
    best = []
    history = [[]] * len(results[0][1])
    for i in results:
        best.append(i[0].cost)
        for idx, elem in enumerate(i[1]):
            history[idx].append(elem.cost)
    return np.mean(best), [np.mean(i) for i in history]


if __name__ == '__main__':
    main()
