import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

FILENAME = Path(__file__).resolve().parent / "results_1586779168" / "save.pkl"

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
    fig, ax = plt.subplots()
    datalen = len(list(data.values())[0])
    for name in names:
        ax.plot(range(datalen), data[name], label=name)

    ax.xaxis.set_ticks(np.arange(0, datalen + 1, 100))
    starty, endy = ax.get_ylim()
    stepy = 20
    ax.yaxis.set_ticks(np.arange(starty - starty%stepy + stepy, endy - stepy, stepy))

    ax.text(0.5, 1.21, "Comparision of tested methods", horizontalalignment='center', fontsize=20,
            transform=ax.transAxes)

    ax.text(0.5, 1.12, "N_ANTS = 20, Alpha = 2, R = 0.25", horizontalalignment='center', fontsize=12,
            transform=ax.transAxes)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
              ncol=3, fancybox=True, shadow=True)
    fig.tight_layout()
    plt.savefig('compare3.svg', dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
