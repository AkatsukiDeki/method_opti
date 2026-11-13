import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# Лабораторная работа №1: Методы одномерного поиска
# Вариант №21: f(x) = x + 1 / (x - 2.5) на отрезке [3, 6]
# Аналитический минимум: x* = 3.5, f(x*) = 4.5
# =====================================================================

VARIANT = 21
A_BOUND = 3.0
B_BOUND = 6.0
X_EXACT = 3.5
F_EXACT = 4.5


def target_f(x):
    """Целевая функция варианта №21: f(x) = x + 1 / (x - 2.5)."""
    return x + 1.0 / (x - 2.5)


def dichotomy_search(f, a, b, delta, eps=None):
    """
    Метод дихотомии для поиска минимума на отрезке [a, b] с точностью delta.
    Условие остановки: полудлина интервала (b - a) / 2 <= delta (или длина b - a <= 2 * delta).
    """
    if eps is None:
        eps = delta * 0.1
    assert 0 < eps < delta, "Параметр eps должен быть положительным и строго меньше delta"

    history_intervals = [b - a]
    history_evals = [0]
    feval_count = 0
    iteration = 0

    while (b - a) / 2.0 > delta:
        iteration += 1
        mid = (a + b) / 2.0
        x1 = mid - eps / 2.0
        x2 = mid + eps / 2.0

        f1, f2 = f(x1), f(x2)
        feval_count += 2

        if f1 < f2:
            b = x2
        else:
            a = x1

        history_intervals.append(b - a)
        history_evals.append(feval_count)

    x_opt = (a + b) / 2.0
    return {
        "name": "Дихотомия",
        "x_opt": x_opt,
        "f_opt": f(x_opt),
        "interval": (a, b),
        "final_interval_len": b - a,
        "achieved_delta": (b - a) / 2.0,
        "abs_error_x": abs(x_opt - X_EXACT),
        "iterations": iteration,
        "fevals": feval_count,
        "history_intervals": history_intervals,
        "history_evals": history_evals,
    }


def golden_section_search(f, a, b, delta):
    """
    Метод золотого сечения для поиска минимума на отрезке [a, b] с точностью delta.
    На первом шаге вычисляется 2 значения функции, на каждом последующем — по 1.
    """
    tau = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.61803398875

    history_intervals = [b - a]
    history_evals = [0]

    if (b - a) <= 2.0 * delta:
        x_opt = (a + b) / 2.0
        return {
            "name": "Золотое сечение",
            "x_opt": x_opt,
            "f_opt": f(x_opt),
            "interval": (a, b),
            "final_interval_len": b - a,
            "achieved_delta": (b - a) / 2.0,
            "abs_error_x": abs(x_opt - X_EXACT),
            "iterations": 0,
            "fevals": 0,
            "history_intervals": history_intervals,
            "history_evals": history_evals,
        }

    x1 = b - tau * (b - a)
    x2 = a + tau * (b - a)
    f1, f2 = f(x1), f(x2)
    feval_count = 2
    iteration = 0

    while True:
        iteration += 1
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            history_intervals.append(b - a)
            history_evals.append(feval_count)
            if (b - a) <= 2.0 * delta:
                break
            x1 = b - tau * (b - a)
            f1 = f(x1)
            feval_count += 1
        else:
            a = x1
            x1 = x2
            f1 = f2
            history_intervals.append(b - a)
            history_evals.append(feval_count)
            if (b - a) <= 2.0 * delta:
                break
            x2 = a + tau * (b - a)
            f2 = f(x2)
            feval_count += 1

    x_opt = (a + b) / 2.0
    return {
        "name": "Золотое сечение",
        "x_opt": x_opt,
        "f_opt": f(x_opt),
        "interval": (a, b),
        "final_interval_len": b - a,
        "achieved_delta": (b - a) / 2.0,
        "abs_error_x": abs(x_opt - X_EXACT),
        "iterations": iteration,
        "fevals": feval_count,
        "history_intervals": history_intervals,
        "history_evals": history_evals,
    }


def fibonacci_search(f, a, b, target_fevals):
    """
    Метод Фибоначчи с фиксированным числом вычислений целевой функции N = target_fevals.
    Используются числа Фибоначчи: F_0 = 0, F_1 = 1, F_2 = 1, F_3 = 2, ..., F_{N+1}.
    За N вычислений функции выполняется (N - 1) сокращений отрезка.
    """
    n = target_fevals
    assert n >= 2, "Количество вычислений функции должно быть не меньше 2"

    fib = [0, 1]
    for _ in range(2, n + 2):
        fib.append(fib[-1] + fib[-2])

    x1 = a + (fib[n - 1] / fib[n + 1]) * (b - a)
    x2 = a + (fib[n] / fib[n + 1]) * (b - a)
    f1, f2 = f(x1), f(x2)
    feval_count = 2
    iteration = 0

    history_intervals = [b - a]
    history_evals = [0]

    # Первые (n - 2) шагов сокращения интервала
    for k in range(1, n - 1):
        iteration += 1
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (fib[n - k - 1] / fib[n - k + 1]) * (b - a)
            if k < n - 2:
                f1 = f(x1)
                feval_count += 1
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + (fib[n - k] / fib[n - k + 1]) * (b - a)
            if k < n - 2:
                f2 = f(x2)
                feval_count += 1

        history_intervals.append(b - a)
        history_evals.append(feval_count)

    # Заключительный (n - 1)-й шаг: точки x1 и x2 совпадают в середине отрезка,
    # поэтому делаем малое смещение eps для последнего (N-го) вычисления функции
    eps = (b - a) * 0.01
    x2 = x1 + eps
    f2 = f(x2)
    feval_count += 1
    iteration += 1

    if f1 < f2:
        b = x2
    else:
        a = x1

    history_intervals.append(b - a)
    history_evals.append(feval_count)

    x_opt = (a + b) / 2.0
    return {
        "name": "Метод Фибоначчи",
        "x_opt": x_opt,
        "f_opt": f(x_opt),
        "interval": (a, b),
        "final_interval_len": b - a,
        "achieved_delta": (b - a) / 2.0,
        "abs_error_x": abs(x_opt - X_EXACT),
        "iterations": iteration,
        "fevals": feval_count,
        "history_intervals": history_intervals,
        "history_evals": history_evals,
    }


def print_comparison_table(delta, results):
    print(f"\nРезультаты для заданной точности Δ = {delta:g} (ε = {delta * 0.1:g} для дихотомии):")
    print("=" * 108)
    header = (
        f"{'Метод':<18} | {'x* (прибл.)':<12} | {'f(x*)':<12} | "
        f"{'Длина b - a':<13} | {'Точность (b-a)/2':<16} | {'|x* - x_exact|':<14} | {'Итер.':<6} | {'Вычисл. f'}"
    )
    print(header)
    print("-" * 108)
    for r in results:
        print(
            f"{r['name']:<18} | {r['x_opt']:<12.7f} | {r['f_opt']:<12.7f} | "
            f"{r['final_interval_len']:<13.5e} | {r['achieved_delta']:<16.5e} | "
            f"{r['abs_error_x']:<14.5e} | {r['iterations']:<6} | {r['fevals']}"
        )
    print("=" * 108)


if __name__ == "__main__":
    # 1. Исследование для нескольких значений точности Δ
    deltas = [1e-1, 1e-2, 1e-4, 1e-6]
    main_results = None

    for d in deltas:
        res_dich = dichotomy_search(target_f, A_BOUND, B_BOUND, delta=d, eps=d * 0.1)
        res_gold = golden_section_search(target_f, A_BOUND, B_BOUND, delta=d)
        res_fib = fibonacci_search(target_f, A_BOUND, B_BOUND, target_fevals=res_gold["fevals"])
        print_comparison_table(d, [res_dich, res_gold, res_fib])
        if d == 1e-4:
            main_results = (res_dich, res_gold, res_fib)

    # 2. Построение графиков: целевая функция и сходимость методов (при Δ = 1e-4)
    res_dich, res_gold, res_fib = main_results

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # График 1: Целевая функция на заданном отрезке [3, 6]
    x_grid = np.linspace(A_BOUND, B_BOUND, 500)
    y_grid = target_f(x_grid)
    axes[0].plot(x_grid, y_grid, color="#1f77b4", lw=2.2, label=r"$f(x) = x + \frac{1}{x - 2.5}$")
    axes[0].scatter([X_EXACT], [F_EXACT], color="crimson", s=90, zorder=5, label=f"Минимум ({X_EXACT}, {F_EXACT})")
    axes[0].axvline(X_EXACT, color="crimson", linestyle="--", alpha=0.5)
    axes[0].set_title("Вариант №21: График функции на отрезке [3, 6]", fontsize=12)
    axes[0].set_xlabel("x", fontsize=11)
    axes[0].set_ylabel("f(x)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=10)

    # График 2: Сходимость длины интервала неопределенности в зависимости от числа вычислений f(x)
    axes[1].plot(res_dich["history_evals"], res_dich["history_intervals"], marker="o", markersize=4, label="Метод дихотомии")
    axes[1].plot(res_gold["history_evals"], res_gold["history_intervals"], marker="s", markersize=4, label="Метод золотого сечения")
    axes[1].plot(res_fib["history_evals"], res_fib["history_intervals"], marker="^", markersize=4, label="Метод Фибоначчи")
    axes[1].set_yscale("log")
    axes[1].set_title(r"Сокращение длины отрезка $(b - a)$ при $\Delta = 10^{-4}$", fontsize=12)
    axes[1].set_xlabel("Количество вычислений целевой функции N", fontsize=11)
    axes[1].set_ylabel("Длина интервала (b - a), лог. шкала", fontsize=11)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("lab1_variant21_plots.png", dpi=150)
    print("\nГрафики успешно сохранены в файл lab1_variant21_plots.png")
