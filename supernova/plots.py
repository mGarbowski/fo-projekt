"""Plots for report and analysis.

Place files in reports/figures
"""

import pandas as pd
import matplotlib.pyplot as plt

from supernova.config import RAW_DATA_DIR


def make_class_distribution_plot():
    meta = pd.read_csv(RAW_DATA_DIR / "training_set_metadata.csv")
    class_counts = meta["target"].value_counts().sort_index()

    plt.figure(figsize=(10, 7))
    plt.bar(class_counts.index.astype(str), class_counts.values)
    plt.title("Rozkład klas", fontsize=18)
    plt.xlabel("Etykieta klasy", fontsize=14)
    plt.ylabel("Liczba wystąpień", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("reports/assets/class_distribution.png", dpi=600)
    plt.close()


if __name__ == "__main__":
    make_class_distribution_plot()
