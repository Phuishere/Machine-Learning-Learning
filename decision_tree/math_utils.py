# entropy.py
from typing import Union
import numpy as np

def self_information(probs: np.ndarray, b: Union[int, float] = 2) -> float:
    """
    Calculate the entropy of a probability distribution.

    Parameters
    ----------
    probs : numpy.array
        An array of probabilities (must sum to 1).
    b : int or str, optional
        The logarithm base.

    Returns
    -------
    float
        The self-information of the distribution.
    """
    probs = probs[probs > 0]
    info = -(np.log(probs) / np.log(b)) # -log_b(probs)
    return info

def entropy_calc(probs: np.ndarray, b: Union[int, float] = 2) -> float:
    """
    Calculate the entropy of a probability distribution.

    Parameters
    ----------
    probs : numpy.array
        An array of probabilities (must sum to 1).
    b : int or str, optional
        The logarithm base. Common choices:
        - 2   : entropy in bits (default).
        - np.e : entropy in nats (natural log).
        - 10  : entropy in bans (log base 10).

    Returns
    -------
    float
        The entropy of the distribution.
    """
    # Self-information
    info = self_information(probs, b = b)
    
    # Mean self-information (entropy)
    weighted_info = probs*info
    return np.sum(weighted_info)

# ==============================

if __name__ == "__main__":
    # Params
    steps = 1000
    step = 1/1000

    # List
    p = []
    for s in range(1, steps + 1):
        p.append(step*s)
    p = np.array(p)
    print(entropy(p))

if False:
    # Visualize
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],  # gọn gàng
        "axes.spines.top": False,   # bỏ viền trên
        "axes.spines.right": False, # bỏ viền phải
        "axes.grid": True,          # bật grid
        "grid.linestyle": "--",     # grid nét đứt mảnh
        "grid.alpha": 0.4,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.frameon": False,    # legend không viền
        "lines.linewidth": 2,       # line dày hơn chút
    })

    # Init ax
    fig, ax = plt.subplots()

    # Check values
    vals = [2, np.e, 5, 10]
    for v in vals:
        # Weighted self info
        si = self_information(p, b=v)
        wsi = si*p

        # Get the line graph
        if v == np.e:
            v = "e"
        ax.plot(p, wsi, label = f"Log cơ số {v}")

    # Plot
    ax.set_xlabel("p", style='italic', fontsize=20)
    ax.set_ylabel("-p × log(p)", labelpad=15, style='italic', fontsize=20)
    ax.set_aspect(1)
    ax.legend(fontsize=14)
    plt.show()

if False:
    P_1 = array([1., 0, 0, 0, 0])
    P_2 = array([0.5, 0.2, 0.1, 0.1, 0.1])
    P_3 = array([0.2, 0.2, 0.2, 0.2, 0.2])
    P_4 = array([0.1, 0.1, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.1])

    # Print out entropy values
    print(entropy(P_1))
    print(entropy(P_2))
    print(entropy(P_3))
    print(entropy(P_4))