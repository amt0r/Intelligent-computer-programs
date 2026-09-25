import matplotlib.pyplot as plt
import numpy as np
from config import target_function, X_MIN, X_MAX, get_display_formula


class Visualizer:
    def plot_results(self, best_max, best_min):
        x_vals = np.linspace(X_MIN, X_MAX, 2000)
        y_vals = [target_function(x) for x in x_vals]

        plt.figure(figsize=(14, 7))
        formula = get_display_formula()
        plt.plot(x_vals, y_vals, color='#3498db', linewidth=1.5, label=f'Y(x) = {formula}')

        x_max = best_max.decode()
        y_max = target_function(x_max)
        plt.scatter([x_max], [y_max], color='#e74c3c', s=150, zorder=5,
                    edgecolors='black', linewidths=1.5, label=f'MAX: x={x_max:.4f}, Y={y_max:.4f}')

        x_min = best_min.decode()
        y_min = target_function(x_min)
        plt.scatter([x_min], [y_min], color='#2ecc71', s=150, zorder=5,
                    edgecolors='black', linewidths=1.5, label=f'MIN: x={x_min:.4f}, Y={y_min:.4f}')

        plt.xlabel('x', fontsize=13)
        plt.ylabel('Y(x)', fontsize=13)
        plt.title(f'Y(x) = {formula}')
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
