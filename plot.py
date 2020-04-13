import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc

plt.style.use('seaborn-whitegrid')

FILENAME = Path(__file__).resolve().parent / "results_13042020" / "save.pkl"

## Config, copied form results_*/tests.py
n_ants = [10, 20, 33]
alphas = [1, 2, 4, 5]
rhos = [0.05, 0.1, 0.25, 0.3]


def main():
    with open(FILENAME, 'rb') as f:
        d = pickle.load(f)
    print(d.keys())

    mutual_params = (20, 2, 0.25)
    names = ["baseline", "two_opt", "candidate_list"]
    data = {
        name: [i for i in d[(name,) + mutual_params][1]] for name in names
    }
    fig, ax = plt.subplots(figsize=(5, 3))
    for name in names:
        print(data[name])
        ax.plot(range(1000), data[name], label=name)

    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
              ncol=2, mode="expand", borderaxespad=0.)
    ax.set_title("Comparision of different methods.\n N_ANTS = 20, Alpha = 2, R = 0.25")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
