import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def cross_val_loss(log_C: float) -> float:
    """Return cross-validated log-loss for LogisticRegression with C=exp(log_C)."""
    C = np.exp(log_C)
    model = LogisticRegression(max_iter=1000, solver="lbfgs", C=C)
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_log_loss")
    return -scores.mean()


def derivative(func, x: float, eps: float = 1e-5) -> float:
    return (func(x + eps) - func(x - eps)) / (2 * eps)


def second_derivative(func, x: float, eps: float = 1e-5) -> float:
    return (func(x + eps) - 2 * func(x) + func(x - eps)) / eps ** 2


def bisection(fprime, a: float, b: float, tol: float = 1e-5, max_iter: int = 100):
    """Bisection method for root finding of derivative."""
    history = []
    fa = fprime(a)
    fb = fprime(b)
    if fa * fb > 0:
        raise ValueError("Derivative has the same sign at interval ends")
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = fprime(c)
        history.append(c)
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c, history
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return c, history


def bracket_root(fprime, a: float, b: float, expand_factor: float = 1.5, max_expand_iter: int = 50):
    """Expand interval until derivative changes sign."""
    fa = fprime(a)
    fb = fprime(b)
    expand_iter = 0
    while fa * fb > 0 and expand_iter < max_expand_iter:
        width = b - a
        a -= 0.5 * (expand_factor - 1) * width
        b += 0.5 * (expand_factor - 1) * width
        fa = fprime(a)
        fb = fprime(b)
        expand_iter += 1
    if fa * fb > 0:
        raise ValueError("Failed to bracket root after expansion attempts")
    return a, b


def secant(fprime, x0: float, x1: float, tol: float = 1e-5, max_iter: int = 100):
    """Secant method for root finding."""
    history = [x0, x1]
    f0 = fprime(x0)
    f1 = fprime(x1)
    for _ in range(max_iter):
        if abs(f1 - f0) < 1e-12:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        history.append(x2)
        if abs(x2 - x1) < tol:
            return x2, history
        x0, x1 = x1, x2
        f0, f1 = f1, fprime(x1)
    return x2, history


def newton(fprime, fsecond, x0: float, tol: float = 1e-5, max_iter: int = 100):
    """Newton's method using derivative and second derivative."""
    history = [x0]
    x = x0
    for _ in range(max_iter):
        fx = fprime(x)
        fxx = fsecond(x)
        if abs(fxx) < 1e-12:
            break
        x_new = x - fx / fxx
        history.append(x_new)
        if abs(x_new - x) < tol:
            return x_new, history
        x = x_new
    return x, history


def hybrid(fprime, fsecond, a: float, b: float, switch_tol: float = 1e-2, tol: float = 1e-5, max_iter: int = 100):
    """Hybrid method combining bisection and Newton."""
    history = []
    c, history_b = bisection(fprime, a, b, tol=switch_tol, max_iter=max_iter)
    history.extend(history_b)
    x, history_n = newton(fprime, fsecond, c, tol=tol, max_iter=max_iter)
    history.extend(history_n[1:])
    return x, history


def main():
    global X, y
    data = load_breast_cancer()
    X, y = data.data, data.target

    def objective(log_C: float) -> float:
        return cross_val_loss(log_C)

    def fprime(log_C: float) -> float:
        return derivative(objective, log_C)

    def fsecond(log_C: float) -> float:
        return second_derivative(objective, log_C)

    a, b = -5.0, 5.0

    # ensure the interval for bisection contains a sign change
    a_bis, b_bis = bracket_root(fprime, a, b)

    x_bis, hist_bis = bisection(fprime, a_bis, b_bis)
    x_sec, hist_sec = secant(fprime, a, b)
    x_new, hist_new = newton(fprime, fsecond, 0.0)
    x_hyb, hist_hyb = hybrid(fprime, fsecond, a_bis, b_bis)

    xs = np.linspace(a, b, 200)
    ys = [objective(x) for x in xs]

    plt.figure()
    plt.plot(xs, ys, label="CV loss")
    plt.axvline(x_bis, color="g", linestyle="--", label="Bisection")
    plt.axvline(x_sec, color="r", linestyle="--", label="Secant")
    plt.axvline(x_new, color="b", linestyle="--", label="Newton")
    plt.axvline(x_hyb, color="k", linestyle="--", label="Hybrid")
    plt.xlabel("log C")
    plt.ylabel("Cross-validation loss")
    plt.legend()
    plt.title("Hyperparameter optimization")
    plt.savefig("optimization_results.png", dpi=150)

    plt.figure()
    plt.plot(hist_bis, label="Bisection")
    plt.plot(hist_sec, label="Secant")
    plt.plot(hist_new, label="Newton")
    plt.plot(hist_hyb, label="Hybrid")
    plt.xlabel("Iteration")
    plt.ylabel("log C")
    plt.legend()
    plt.title("Convergence history")
    plt.savefig("convergence_history.png", dpi=150)


if __name__ == "__main__":
    main()
