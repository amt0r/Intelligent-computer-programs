import math


FORMULA_STR = "(1/x) * math.cos(x**2 + 1/x)"


def target_function(x):
    return eval(FORMULA_STR)


def get_display_formula():
    return FORMULA_STR


X_MIN = 1
X_MAX = 10

E = 0.0001 # Точність


def compute_num_bits(x_min, x_max, e):
    required = (x_max - x_min) / e
    nb = 1
    while (2 ** nb - 1) < required:
        nb += 1
    return nb

# 17
NB = compute_num_bits(X_MIN, X_MAX, E)