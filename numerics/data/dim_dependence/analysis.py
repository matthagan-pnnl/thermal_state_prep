#!/usr/bin/env python
"""Plot interaction count (gamma) vs. Hilbert space dimension."""
import json
from os.path import dirname, join, splitext

import matplotlib.pyplot as plt
import numpy as np

FILENAME = "sho_interactions_vs_dim_beta_1.json"


def power_law_fit(x, y):
    """Fit y = A * x^p via linear regression in log-log space.

    Returns (p, A, r_squared) so the reported slope is an actual fit to
    the data rather than an assumed/rounded value.
    """
    log_x = np.log(x)
    log_y = np.log(y)
    p, log_A = np.polyfit(log_x, log_y, 1)
    fit_log_y = p * log_x + log_A
    ss_res = np.sum((log_y - fit_log_y) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    return p, np.exp(log_A), r_squared


def plot_dim_dependence(filepath):
    with open(filepath) as f:
        data = json.load(f)

    dims = np.array(data["dims"])
    series = {
        "Uniform Sampling": np.array(data["uniform_gamma"]),
        "Sample From Spectrum": np.array(data["spectral_gamma"]),
    }

    fig, ax = plt.subplots()
    for label, gamma in series.items():
        p, _, r_squared = power_law_fit(dims, gamma)
        ax.plot(
            dims,
            gamma,
            marker="o",
            label=f"{label} (fit slope = {p:.2f}, $R^2$ = {r_squared:.3f})",
        )

    ax.set_xlabel("Hilbert space dimension")
    ax.set_ylabel(r"Number of interactions, $\Gamma$")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()

    root, _ = splitext(filepath)
    out_path = root + "_linear_x.pdf"
    fig.savefig(out_path)
    print(f"saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_dim_dependence(join(dirname(__file__), FILENAME))
