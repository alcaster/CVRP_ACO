import time
from pathlib import Path
from typing import Dict, Tuple
import pickle
from aco_tsp_reworked import Config, run_aco, Solution


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

    n_ants = [20, 40, 100, 200]
    alphas = [1, 2, 4, 5]
    rhos = [0.1, 0.2, 0.4, 0.5]

    timestamp = int(time.time())
    log_dir = Path(__file__).resolve().parent / f"results_{timestamp}"
    log_dir.mkdir()

    results: Dict[Tuple, Solution] = {}
    for name, config in configs.items():
        print(f"Testing config: {name}")
        for ants in n_ants:
            print(f"Ants: {ants}")
            config.NUM_ANTS = ants
            for alpha in alphas:
                print(f"Alpha: {alpha}")
                config.ALPHA = alpha
                for rho in rhos:
                    print(f"Rho: {rho}")
                    config.RHO = rho
                    result = run_aco(config)
                    with open(log_dir / "log.txt", 'a') as f:
                        f.write(f"{name.upper()}: ants: {ants}, alpha: {alpha}, rho: {rho} = {result.cost}\n")
                    results[name, ants, alpha, rho] = result

    with open(log_dir / "save.pkl", 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
