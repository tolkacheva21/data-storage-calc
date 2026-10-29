"""
Модель системной динамики характеристик эффективности ПО хранилища данных.

Реализует уравнения (2.9)-(2.21) из документа «Моделирование характеристик
хранилищ данных»:

    dX_i(t, p(t))/dt = 1/max X_i* · ( произведение f_k(X_j(t)) ... комбинация ζ_m(t) )

где X_i, i = 1..14  - характеристики эффективности (выходные переменные),
    ζ_m, m = 1..5   - возмущения со значениями из [0, 1],
    f_k             - вспомогательные функции (в документе: «полиномы невысокой
                      степени») - здесь задаются как f(x) = a0 + a1·x + a2·x² + a3·x³.

Чего в документе НЕТ и что добавлено в код как настраиваемые допущения:
  * вид и коэффициенты функций f_k            -> a0, a1, a2, a3 (редактируются);
  * временной масштаб модели (tau)            -> множитель правой части;
  * вид целевой функции Z_p                   -> взвешенный квадрат нормированного
                                                 отклонения от рекомендованных X_i*;
  * вид влияния плана мероприятий p(t)        -> добавка u_i к скорости изменения X_i;
  * временные профили возмущений ζ_m(t).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

N_X = 14
N_Z = 5

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x / 2.x

X_NAMES: List[str] = [
    "Эффективность функционирования хранилища данных",
    "Качество ПО",
    "Корректность ПО",
    "Надёжность ПО",
    "Доступность ПО",
    "Возможность интенсивного использования ПО",
    "Прослеживаемость ПО",
    "Функциональная полнота ПО",
    "Обеспечение требуемой последовательности работ",
    "Практичность ПО",
    "Устойчивость к ошибкам данных",
    "Эффективность выполнения транзакций",
    "Мотивация персонала",
    "Удобство тестирования ПО",
]

X_SHORT: List[str] = [
    "X1 Эффективность ХД",
    "X2 Качество ПО",
    "X3 Корректность",
    "X4 Надёжность",
    "X5 Доступность",
    "X6 Интенсивное использ.",
    "X7 Прослеживаемость",
    "X8 Функц. полнота",
    "X9 Послед. работ",
    "X10 Практичность",
    "X11 Устойч. к ошибкам",
    "X12 Транзакции",
    "X13 Мотивация персонала",
    "X14 Удобство тестирования",
]

ZETA_NAMES: List[str] = [
    "ζ1 Рост числа источников данных",
    "ζ2 Частота смены сроков отчётности",
    "ζ3 Сокращение поддержки вендора",
    "ζ4 Рост перехода на Open Source",
    "ζ5 Рост числа новых стандартов Open Source",
]


# --------------------------------------------------------------------------- #
#  Структура уравнений (прочитана с формул документа)
# --------------------------------------------------------------------------- #
# Каждый множитель - пара (номер функции f_k в документе, номер аргумента j),
# т.е. f_k(X_j(t)).  Номера j нумеруются с 1.
#
# kind:
#   "std"     : P·Σ ζ[plus]  -  Q·Σ ζ[minus]
#   "sub"     : P            -  Q·Σ ζ[minus]
#   "diffmul" : (P - Q)·Σ ζ[plus]
#   "x4"      : P·ζ5 - (ζ1+ζ2+ζ3+ζ4 + Q)           (уравнение 2.12, см. README)
# P - произведение основных множителей, Q - произведение «вторых» множителей
# (если Q пуст, он равен 1).
@dataclass(frozen=True)
class EqSpec:
    p: Tuple[Tuple[int, int], ...]
    kind: str = "std"
    q: Tuple[Tuple[int, int], ...] = ()
    plus: Tuple[int, ...] = ()      # номера ζ (с 1)
    minus: Tuple[int, ...] = ()


def _seq(start: int, js: Sequence[int]) -> Tuple[Tuple[int, int], ...]:
    """f_start(X_{js[0]}), f_{start+1}(X_{js[1]}), ..."""
    return tuple((start + n, j) for n, j in enumerate(js))


_ALL = tuple(range(1, 15))

EQUATIONS: Dict[int, EqSpec] = {
    1: EqSpec(_seq(1, _ALL), "std", plus=(1, 2, 3), minus=(4, 5)),                      # (2.9)
    2: EqSpec(_seq(15, _ALL), "std", plus=(1, 2, 3, 4), minus=(5,)),                    # (2.10)
    3: EqSpec(_seq(29, _ALL), "std", plus=(1, 2, 3, 4), minus=(5,)),                    # (2.11)
    4: EqSpec(((43, 1), (44, 2), (45, 3), (46, 4), (49, 7), (50, 8), (51, 9), (52, 10),
               (53, 11), (54, 12), (55, 13), (56, 14)),
              "x4", q=((47, 5), (48, 6))),                                              # (2.12)
    5: EqSpec(((57, 4), (58, 6), (59, 9), (60, 10), (61, 13)),
              "std", plus=(1, 2, 4, 5), minus=(5,)),                                    # (2.13)
    6: EqSpec(_seq(62, _ALL), "std", plus=(1, 2), minus=(5,)),                          # (2.14)
    7: EqSpec(((76, 2), (77, 4), (78, 14)), "sub", minus=(5,)),                         # (2.14')
    8: EqSpec(((76, 1), (77, 2), (78, 3), (79, 4), (80, 6), (81, 9), (82, 10)),
              "std", plus=(1, 2, 3), minus=(1, 2)),                                     # (2.15)
    9: EqSpec(_seq(83, (1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14)),
              "std", plus=(4, 5), minus=(1, 2, 3)),                                     # (2.17)
    10: EqSpec(_seq(95, tuple(range(1, 13))), "std", q=((107, 13), (108, 14)),
               plus=(1, 2), minus=(3,)),                                                # (2.17')
    11: EqSpec(_seq(109, (1, 2, 3, 4, 8, 10, 12, 13, 14)), "sub",
               q=((118, 5), (119, 6)), minus=(1, 2, 3, 4, 5)),                          # (2.19): P - Q*(z1+..+z5)
    12: EqSpec(_seq(120, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14)),
               "std", plus=(1, 4, 5), minus=(3,)),                                      # (2.20)
    13: EqSpec(_seq(133, (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14)), "sub",
               q=((146, 9),), minus=(1, 2, 3, 4, 5)),                                   # (2.21)
    14: EqSpec(_seq(147, (5, 7, 11, 13)), "sub", minus=(1, 2, 3, 4, 5)),                # (2.21')
}

# Функции, которые в документе прямо заданы равными единице: (i, j)
FIXED_ONE = {(1, 1), (2, 2), (6, 6), (10, 10)}

# Коэффициенты полинома f(x) = a0 + a1*x + a2*x^2 по умолчанию
DEFAULT_POLY = (0.9, 0.2, 0.0, 0.0)   # a0, a1, a2, a3: f(x) = a0 + a1·x + a2·x² + a3·x³


def f_table_rows() -> List[dict]:
    """Список всех функций f модели: уравнение i, аргумент j, номер f в документе."""
    rows = []
    for i, eq in EQUATIONS.items():
        for part, items in (("P", eq.p), ("Q", eq.q)):
            for k, j in items:
                rows.append({"i": i, "j": j, "f": k, "part": part})
    return rows


# --------------------------------------------------------------------------- #
#  Конфигурация
# --------------------------------------------------------------------------- #
def default_config() -> dict:
    zeta = [
        {"kind": "ramp",  "v0": 0.10, "v1": 0.50, "t0": 0.0, "t1": 1.0},
        {"kind": "const", "v0": 0.20, "v1": 0.20, "t0": 0.0, "t1": 1.0},
        {"kind": "step",  "v0": 0.00, "v1": 0.60, "t0": 0.5, "t1": 1.0},
        {"kind": "sin",   "v0": 0.30, "v1": 0.20, "t0": 0.0, "t1": 1.0},
        {"kind": "const", "v0": 0.10, "v1": 0.10, "t0": 0.0, "t1": 1.0},
    ]
    f_coefs = {}
    for r in f_table_rows():
        key = f"{r['i']},{r['j']}"
        f_coefs[key] = [1.0, 0.0, 0.0, 0.0] if (r["i"], r["j"]) in FIXED_ONE else list(DEFAULT_POLY)
    return {
        "T": 1.0,
        "dt": 0.05,
        "tau": 0.1,
        # начальные значения X(0) - как в примере из методички, X* = 1
        "indicators": [
            {"x0": x0, "xstar": 1.0, "xmin": 0.0, "xmax": 1.0, "gamma": 1.0}
            for x0 in (0.8, 0.7, 0.83, 0.72, 0.6, 0.52, 0.48, 0.22, 0.27, 0.54, 0.2, 0.4, 0.32, 0.14)
        ],
        "zeta": zeta,
        "f_coefs": f_coefs,
        "plan": {"start": 0.0, "u": [0.0] * N_X},
    }


# --------------------------------------------------------------------------- #
#  Допустимые диапазоны входных данных и их проверка (валидация)
# --------------------------------------------------------------------------- #
# Границы (min, max) включительно. Интерфейс берёт их отсюда же, чтобы показать
# пользователю допустимые значения, а проверка на «бэке» (validate_config) срабатывает
# независимо от интерфейса - в том числе внутри simulate().
LIMITS = {
    "T": (0.1, 1.0),          # горизонт моделирования: время t от 0 до T, по заданию не больше 1
    "dt": (0.001, 1.0),       # шаг интегрирования
    "tau": (0.001, 1.0),      # временной масштаб
    "x": (0.0, 1.0),          # Xmin, Xmax, X(0)
    "xstar": (0.01, 1.0),     # рекомендованные значения X* (делитель в уравнениях, поэтому не ноль)
    "gamma": (0.0, 1.0),      # веса показателей в целевой функции
    "zeta": (0.0, 1.0),       # параметры v0, v1 возмущений
    "t": (0.0, 1.0),          # параметры t0, t1 возмущений и момент начала плана
    "coef": (0.0, 1.0),       # коэффициенты a0..a3 полиномов f
    "u": (0.0, 1.0),          # добавки плана мероприятий
}
# Минимум шагов интегрирования: если пользователь задал слишком крупный Δt, расчёт сам делает шаг
# мельче, чтобы на отрезке 0…T было достаточно точек (график и «сечения по времени» не вырождаются).
MIN_STEPS = 20


class ConfigError(ValueError):
    """Некорректные входные данные; список сообщений - в атрибуте problems."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _num(v):
    """float(v), если это конечное число, иначе None (пусто, текст, NaN, inf, bool)."""
    if isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _get(d, key):
    return d.get(key) if isinstance(d, dict) else None


def validate_config(cfg: dict) -> List[str]:
    """Проверка входных данных: числа, попадающие в допустимые диапазоны LIMITS.

    Возвращает список понятных сообщений об ошибках (пустой - данные верны).
    Других условий нет, кроме одного: Xmax должен быть больше Xmin."""
    try:
        ind, zeta, plan, fco = cfg["indicators"], cfg["zeta"], cfg["plan"], cfg["f_coefs"]
        _ = cfg["T"], cfg["dt"], cfg["tau"], plan["start"], plan["u"]
        if len(ind) != N_X or len(zeta) != N_Z or len(plan["u"]) != N_X:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return ["Некорректная структура входных данных (ожидается 14 показателей, 5 возмущений и 14 значений плана)."]

    p: List[str] = []

    def chk(label, value, lo, hi):
        x = _num(value)
        if x is None:
            p.append(f"{label}: нужно число (поле пустое или содержит недопустимое значение)")
            return None
        if x < lo or x > hi:
            p.append(f"{label}: значение {x:g} вне допустимого диапазона [{lo:g}; {hi:g}]")
            return None
        return x

    # ---- общие параметры
    chk("Горизонт T (время t от 0 до T)", cfg["T"], *LIMITS["T"])
    chk("Шаг интегрирования Δt", cfg["dt"], *LIMITS["dt"])
    chk("Временной масштаб τ", cfg["tau"], *LIMITS["tau"])

    # ---- показатели
    for n, d in enumerate(ind, start=1):
        name = f"X{n}"
        xmin = chk(f"{name}, Xmin", _get(d, "xmin"), *LIMITS["x"])
        xmax = chk(f"{name}, Xmax", _get(d, "xmax"), *LIMITS["x"])
        if xmin is not None and xmax is not None and xmax <= xmin:
            p.append(f"{name}: Xmax ({xmax:g}) должен быть больше Xmin ({xmin:g})")
        chk(f"{name}, начальное значение X(0)", _get(d, "x0"), *LIMITS["x"])
        chk(f"{name}, рекомендованное значение X*", _get(d, "xstar"), *LIMITS["xstar"])
        chk(f"{name}, вес γ", _get(d, "gamma"), *LIMITS["gamma"])

    # ---- возмущения
    for m, z in enumerate(zeta, start=1):
        label = f"ζ{m}"
        if _get(z, "kind") not in ZETA_KINDS:
            p.append(f"{label}: неизвестный вид возмущения «{_get(z, 'kind')}»")
            continue
        chk(f"{label}, v0", _get(z, "v0"), *LIMITS["zeta"])
        chk(f"{label}, v1", _get(z, "v1"), *LIMITS["zeta"])
        chk(f"{label}, t0", _get(z, "t0"), *LIMITS["t"])
        chk(f"{label}, t1", _get(z, "t1"), *LIMITS["t"])

    # ---- функции f
    bad = []
    for r in f_table_rows():
        coefs = fco.get(f"{r['i']},{r['j']}") if isinstance(fco, dict) else None
        ok = isinstance(coefs, (list, tuple)) and len(coefs) >= 1 and all(
            _num(c) is not None and LIMITS["coef"][0] <= _num(c) <= LIMITS["coef"][1] for c in coefs)
        if not ok:
            bad.append(f"f{r['f']}(X{r['j']})")
    if bad:
        shown = ", ".join(bad[:8]) + (f" и ещё {len(bad) - 8}" if len(bad) > 8 else "")
        p.append(f"Коэффициенты a0…a3 функций должны быть числами в диапазоне [{LIMITS['coef'][0]:g}; "
                 f"{LIMITS['coef'][1]:g}]. Ошибки в: {shown}")

    # ---- план мероприятий
    chk("План мероприятий, момент начала t_старт", plan["start"], *LIMITS["t"])
    bad_u = [f"X{n}" for n, u in enumerate(plan["u"], start=1)
             if _num(u) is None or not (LIMITS["u"][0] <= _num(u) <= LIMITS["u"][1])]
    if bad_u:
        p.append(f"План мероприятий: значения uᵢ должны быть числами в диапазоне [{LIMITS['u'][0]:g}; "
                 f"{LIMITS['u'][1]:g}]. Ошибки в: {', '.join(bad_u)}")
    return p


def _bounds(d: dict) -> Tuple[float, float]:
    """Границы показателя для генерации случайных значений (при некорректных - [0, 1])."""
    lo, hi = _num(_get(d, "xmin")), _num(_get(d, "xmax"))
    xlo, xhi = LIMITS["x"]
    if lo is None or hi is None:
        return xlo, xhi
    lo, hi = max(lo, xlo), min(hi, xhi)
    return (lo, hi) if hi > lo else (xlo, xhi)


# --------------------------------------------------------------------------- #
#  Генерация случайных данных (кнопки «Рандомизировать»)
# --------------------------------------------------------------------------- #
def randomize_x0(cfg: dict, rng: np.random.Generator) -> None:
    """Начальные значения X_i(0): случайные из [Xmin, Xmax]."""
    for d in cfg["indicators"]:
        lo, hi = _bounds(d)
        d["x0"] = round(float(rng.uniform(lo, hi)), 2)


def randomize_xstar(cfg: dict, rng: np.random.Generator) -> None:
    """Рекомендованные значения X_i*: из верхней половины диапазона, строго больше нуля."""
    for d in cfg["indicators"]:
        lo, hi = _bounds(d)
        d["xstar"] = max(round(float(rng.uniform(lo + 0.5 * (hi - lo), hi)), 2), LIMITS["xstar"][0])


def randomize_zeta(cfg: dict, rng: np.random.Generator) -> None:
    """Случайные профили возмущений ζ_m(t) на горизонте [0, T]."""
    T = _num(cfg.get("T")) or 1.0
    kinds = list(ZETA_KINDS)
    for spec in cfg["zeta"]:
        kind = str(rng.choice(kinds))
        t0 = round(float(rng.uniform(0.0, 0.5 * T)), 2)
        t1 = round(float(rng.uniform(0.5 * T, T)), 2)
        v0 = round(float(rng.uniform(0.0, 0.6)), 2)
        v1 = round(float(rng.uniform(0.0, 0.3 if kind == "sin" else 1.0)), 2)
        if kind == "const":
            v1 = v0
        spec.update({"kind": kind, "v0": v0, "v1": v1, "t0": t0, "t1": t1})


def randomize_f(cfg: dict, rng: np.random.Generator) -> None:
    """Случайные коэффициенты полиномов f (кроме функций, равных 1)."""
    for key in cfg["f_coefs"]:
        i, j = (int(s) for s in key.split(","))
        if (i, j) not in FIXED_ONE:
            cfg["f_coefs"][key] = [round(float(rng.uniform(0.5, 1.0)), 2),
                                   round(float(rng.uniform(0.0, 0.5)), 2), 0.0, 0.0]

# --------------------------------------------------------------------------- #
#  Возмущения ζ(t)
# --------------------------------------------------------------------------- #
ZETA_KINDS = {
    "const": "Постоянное",
    "step": "Ступенька",
    "ramp": "Линейный рост/спад",
    "sin": "Синусоида",
    "pulse": "Импульс",
}


def zeta_profile(spec: dict, t: np.ndarray) -> np.ndarray:
    """Значения одного возмущения ζ(t) на сетке t, обрезанные в [0, 1].

    const : v0
    step  : v0 при t < t0, иначе v1
    ramp  : линейно от v0 (в t0) до v1 (в t1), вне отрезка - константа
    sin   : v0 + v1·sin(2π (t - t0)/t1), t1 - период
    pulse : v1 при t0 <= t <= t1, иначе v0
    """
    kind = spec["kind"]
    v0, v1, t0, t1 = (float(spec[k]) for k in ("v0", "v1", "t0", "t1"))
    t = np.asarray(t, dtype=float)
    if kind == "const":
        y = np.full_like(t, v0)
    elif kind == "step":
        y = np.where(t < t0, v0, v1)
    elif kind == "ramp":
        span = t1 - t0
        w = np.clip((t - t0) / span, 0.0, 1.0) if span > 0 else (t >= t0).astype(float)
        y = v0 + (v1 - v0) * w
    elif kind == "sin":
        period = t1 if t1 > 0 else 1.0
        y = v0 + v1 * np.sin(2.0 * np.pi * (t - t0) / period)
    elif kind == "pulse":
        y = np.where((t >= t0) & (t <= t1), v1, v0)
    else:
        raise ValueError(f"Неизвестный вид возмущения: {kind}")
    return np.clip(y, 0.0, 1.0)


# --------------------------------------------------------------------------- #
#  Подготовка матриц и правая часть
# --------------------------------------------------------------------------- #
class _Compiled:
    """Конфигурация, разложенная в матрицы для быстрого векторного расчёта."""

    def __init__(self, cfg: dict):
        self.A = np.zeros((4, N_X, N_X))            # коэффициенты a0..a3 для f(i, j)
        self.Pm = np.zeros((N_X, N_X), dtype=bool)  # маска множителей P
        self.Qm = np.zeros((N_X, N_X), dtype=bool)  # маска множителей Q
        self.Sp = np.zeros((N_X, N_Z))              # ζ в «плюс»-группе
        self.Sm = np.zeros((N_X, N_Z))              # ζ в «минус»-группе
        self.kind = np.zeros(N_X, dtype=int)        # 0 std, 1 sub, 2 diffmul, 3 x4
        kinds = {"std": 0, "sub": 1, "diffmul": 2, "x4": 3}

        for i, eq in EQUATIONS.items():
            r = i - 1
            self.kind[r] = kinds[eq.kind]
            for _, j in eq.p:
                self.Pm[r, j - 1] = True
            for _, j in eq.q:
                self.Qm[r, j - 1] = True
            for m in eq.plus:
                self.Sp[r, m - 1] = 1.0
            for m in eq.minus:
                self.Sm[r, m - 1] = 1.0
            if eq.kind == "x4":                      # P·ζ5 - (ζ1+ζ2+ζ3+ζ4 + Q)
                self.Sp[r, 4] = 1.0
                self.Sm[r, :4] = 1.0
        for key, coefs in cfg["f_coefs"].items():
            i, j = (int(s) for s in key.split(","))
            for n in range(min(len(coefs), 4)):
                self.A[n, i - 1, j - 1] = float(coefs[n])

        ind = cfg["indicators"]
        self.xmin = np.array([float(d["xmin"]) for d in ind])
        self.xmax = np.array([float(d["xmax"]) for d in ind])
        self.xstar = np.array([float(d["xstar"]) for d in ind])
        self.x0 = np.array([float(d["x0"]) for d in ind])
        self.gamma = np.array([float(d["gamma"]) for d in ind])
        self.norm = 1.0 / np.where(np.abs(self.xstar) > 1e-12, self.xstar, 1.0)
        self.tau = float(cfg["tau"])
        self.u = np.array(cfg["plan"]["u"], dtype=float)
        self.u_start = float(cfg["plan"]["start"])

    def rhs(self, x: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        xx = x[None, :]
        F = self.A[0] + self.A[1] * xx + self.A[2] * xx ** 2 + self.A[3] * xx ** 3
        P = np.where(self.Pm, F, 1.0).prod(axis=1)
        Q = np.where(self.Qm, F, 1.0).prod(axis=1)
        sp = self.Sp @ z
        sm = self.Sm @ z
        v_std = P * sp - Q * sm
        v_sub = P - Q * sm
        v_dif = (P - Q) * sp
        v_x4 = P * z[4] - (sm + Q)
        val = np.choose(self.kind, [v_std, v_sub, v_dif, v_x4])
        dx = self.tau * self.norm * val
        if t >= self.u_start:
            dx = dx + self.u
        return dx


@dataclass
class Result:
    t: np.ndarray                 # (n,)
    X: np.ndarray                 # (n, 14)
    zeta: np.ndarray              # (n, 5)
    Z: np.ndarray                 # (n,) целевая функция в каждый момент
    cfg: dict = field(repr=False, default_factory=dict)

    @property
    def Z_final(self) -> float:
        return float(self.Z[-1])

    @property
    def Z_mean(self) -> float:
        return float(_trapz(self.Z, self.t) / (self.t[-1] - self.t[0])) if len(self.t) > 1 else float(self.Z[0])


def objective(X: np.ndarray, comp: _Compiled) -> np.ndarray:
    """Z = Σ γ_i · ((X_i - X_i*) / (X_i^max - X_i^min))²,  Σγ_i нормируется к 1.

    X может быть вектором (14,) или матрицей (n, 14)."""
    rng = np.where(np.abs(comp.xmax - comp.xmin) > 1e-12, comp.xmax - comp.xmin, 1.0)
    g = comp.gamma / comp.gamma.sum() if comp.gamma.sum() > 0 else np.full(N_X, 1.0 / N_X)
    d = (X - comp.xstar) / rng
    return (g * d * d).sum(axis=-1)


# --------------------------------------------------------------------------- #
#  Интегрирование
# --------------------------------------------------------------------------- #
def simulate(cfg: dict, zeta_override: Dict[int, float] | None = None) -> Result:
    """Расчёт X_i(t) методом Рунге-Кутты 4-го порядка с ограничением X_i в [Xmin, Xmax].

    zeta_override: {номер ζ (с 1): постоянное значение} - заменяет профиль возмущения
    (нужно для анализа чувствительности)."""
    problems = validate_config(cfg)
    for m, v in (zeta_override or {}).items():
        if m not in range(1, N_Z + 1) or _num(v) is None or not (0.0 <= _num(v) <= 1.0):
            problems.append(f"Возмущение ζ{m}: уровень должен быть числом в диапазоне [0; 1]")
    if problems:
        raise ConfigError(problems)
    comp = _Compiled(cfg)
    T, dt = float(cfg["T"]), float(cfg["dt"])
    n = max(int(round(T / dt)), MIN_STEPS)
    h = T / n
    t = np.linspace(0.0, T, n + 1)
    t_half = np.linspace(0.0, T, 2 * n + 1)          # для промежуточных стадий RK4

    zh = np.zeros((2 * n + 1, N_Z))
    for m in range(N_Z):
        if zeta_override and (m + 1) in zeta_override:
            zh[:, m] = np.clip(zeta_override[m + 1], 0.0, 1.0)
        else:
            zh[:, m] = zeta_profile(cfg["zeta"][m], t_half)

    X = np.zeros((n + 1, N_X))
    x = np.clip(comp.x0, comp.xmin, comp.xmax)
    X[0] = x
    for k in range(n):
        tk = t[k]
        z0, z1, z2 = zh[2 * k], zh[2 * k + 1], zh[2 * k + 2]
        k1 = comp.rhs(x, z0, tk)
        k2 = comp.rhs(np.clip(x + 0.5 * h * k1, comp.xmin, comp.xmax), z1, tk + 0.5 * h)
        k3 = comp.rhs(np.clip(x + 0.5 * h * k2, comp.xmin, comp.xmax), z1, tk + 0.5 * h)
        k4 = comp.rhs(np.clip(x + h * k3, comp.xmin, comp.xmax), z2, tk + h)
        x = np.clip(x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, comp.xmin, comp.xmax)
        X[k + 1] = x

    return Result(t=t, X=X, zeta=zh[::2], Z=objective(X, comp), cfg=copy.deepcopy(cfg))


def sensitivity(cfg: dict, zeta_index: int, levels: Sequence[float]) -> Dict[str, np.ndarray]:
    """Зависимость конечных значений X_i(T) и Z(T) от уровня возмущения ζ_m (m = zeta_index, с 1).

    Остальные возмущения берутся из конфигурации без изменений."""
    problems = validate_config(cfg)
    if problems:
        raise ConfigError(problems)
    fast = copy.deepcopy(cfg)
    fast["dt"] = max(float(cfg["dt"]), float(cfg["T"]) / 200.0)   # грубее шаг - быстрее
    xs, zs = [], []
    for lv in levels:
        r = simulate(fast, {zeta_index: float(lv)})
        xs.append(r.X[-1])
        zs.append(r.Z_final)
    return {"levels": np.asarray(levels, dtype=float), "X_final": np.array(xs), "Z_final": np.array(zs)}


# --------------------------------------------------------------------------- #
#  Матрица взаимовлияния (знаковая): строки X1..X14, столбцы X1..X14 и ζ1..ζ5
# --------------------------------------------------------------------------- #
def influence_sign_matrix(cfg: dict) -> np.ndarray:
    """Знаковая матрица влияния размера 14 x 19 со значениями 1, -1, 0.

    Строка i - показатель X_i (его уравнение). Столбцы: X_1..X_14 (влияние показателя X_j на
    скорость изменения X_i), затем ζ_1..ζ_5 (влияние возмущений).
      1  - фактор входит в «ростовую» часть уравнения (его увеличение ускоряет рост X_i);
     -1  - фактор входит в «убывающую» часть (его увеличение замедляет рост X_i);
      0  - в уравнение X_i не входит (или функция f постоянна, как f ≡ 1 по документу).
    Для X_j знак определяется положением функции f_k(X_j) в уравнении (множитель P - плюс, Q - минус),
    для ζ_m - знаком коэффициента при ζ_m в правой части при начальных значениях X(0)."""
    problems = validate_config(cfg)
    if problems:
        raise ConfigError(problems)
    S = np.zeros((N_X, N_X + N_Z), dtype=int)
    for i, eq in EQUATIONS.items():
        for sign, items in ((1, eq.p), (-1, eq.q)):
            for _, j in items:
                coefs = cfg["f_coefs"][f"{i},{j}"]
                if any(abs(float(c)) > 1e-12 for c in list(coefs)[1:4]):   # f зависит от X_j
                    S[i - 1, j - 1] = sign
    comp = _Compiled(cfg)
    x = np.clip(comp.x0, comp.xmin, comp.xmax)
    base0 = comp.rhs(x, np.zeros(N_Z), 0.0)
    for m in range(N_Z):
        e = np.zeros(N_Z)
        e[m] = 1.0
        coef = comp.rhs(x, e, 0.0) - base0          # правая часть линейна по ζ, поэтому это точный коэффициент
        S[:, N_X + m] = np.where(np.abs(coef) < 1e-12, 0, np.sign(coef)).astype(int)
    return S


# --------------------------------------------------------------------------- #
#  Запись уравнений в том виде, как в документе (для отображения в интерфейсе)
# --------------------------------------------------------------------------- #
# Номера уравнений как в документе (в документе (2.14), (2.17), (2.21) встречаются дважды)
DOC_NUMBERS = {1: "2.9", 2: "2.10", 3: "2.11", 4: "2.12", 5: "2.13", 6: "2.14", 7: "2.14",
               8: "2.15", 9: "2.17", 10: "2.17", 11: "2.19", 12: "2.20", 13: "2.21", 14: "2.21"}

# Подписи к уравнениям - дословно из документа
DOC_DESCRIPTIONS = {
    1: "эффективности функционирования хранилища данных",
    2: "качества программного обеспечения",
    3: "корректности программного обеспечения",
    4: "надежности программного обеспечения",
    5: "доступности программного обеспечения",
    6: "возможности интенсивного использования программного обеспечения",
    7: "прослеживаемости программного обеспечения",
    8: "функциональной полноты программного обеспечения",
    9: "обеспечение требуемой последовательности работ при проектировании хранилища",
    10: "практичности программного обеспечения",
    11: "устойчивости к ошибкам данных программного обеспечения",
    12: "эффективности выполнения транзакций",
    13: "степени мотивации персонала, осуществляющего эксплуатацию хранилища данных",
    14: "удобства тестирования программного обеспечения",
}

# Строки под уравнениями, которые есть в документе (диапазоны номеров функций)
DOC_NOTES = {3: r"f_{i},\ i=\overline{28,42}", 14: r"f_{i},\ i=\overline{147,150}"}

_FACTORS_PER_LINE = 6


def _lx_f(k: int, j: int) -> str:
    return f"f_{{{k}}}(X_{{{j}}}(t))"


def _lx_z(m: int) -> str:
    return f"\\zeta_{{{m}}}(t)"


def _lx_zsum(ms: Sequence[int]) -> str:
    return "(" + "+".join(_lx_z(m) for m in ms) + ")"


def _lx_prod(items) -> str:
    return r"\,".join(_lx_f(k, j) for k, j in items)


def f_value(cfg: dict, i: int, j: int, xj: float) -> float:
    """Значение функции f в уравнении X_i при аргументе X_j = xj: a0 + a1·x + a2·x² + a3·x³
    (коэффициенты берутся из cfg["f_coefs"], то есть из вкладки «Функции fₖ»)."""
    c = [float(v) for v in cfg["f_coefs"][f"{i},{j}"]] + [0.0] * 4
    return sum(c[n] * xj ** n for n in range(4))


def equation_latex(i: int, f_repr=None, z_repr=None, prefix: str | None = None) -> str:
    """Уравнение для X_i в виде LaTeX.

    Без аргументов - символьная запись как в документе. Если переданы f_repr(k, j) и z_repr(m)
    (функции, возвращающие строку с числом), то f_k(X_j(t)) и ζ_m(t) заменяются числами,
    а множители соединяются знаком умножения. prefix - множитель перед скобкой
    (по умолчанию 1/max X_i*)."""
    numeric = f_repr is not None
    fr = f_repr or _lx_f
    zr = z_repr or _lx_z
    sep = r" \cdot " if numeric else r"\,"          # между множителями-функциями
    brk = r"\cdot" if numeric else r"\star"          # знак при переносе строки

    def zsum(ms):
        return "(" + "+".join(zr(m) for m in ms) + ")"

    def zone(ms):
        return zsum(ms) if len(ms) > 1 else zr(ms[0])

    eq = EQUATIONS[i]
    P = [fr(k, j) for k, j in eq.p]
    Qs = sep.join(fr(k, j) for k, j in eq.q)
    q_wrapped = f"({Qs})" if len(eq.q) > 1 else Qs

    if eq.kind == "std":
        tail = zsum(eq.plus)
        if eq.minus:
            if eq.q:
                tail += " - " + q_wrapped + zone(eq.minus)
            else:
                tail += "".join(f" - {zr(m)}" for m in eq.minus)
    elif eq.kind == "sub":
        tail = "- " + (q_wrapped if eq.q else "") + zone(eq.minus)
    elif eq.kind == "diffmul":
        tail = f"- {q_wrapped}"
    else:  # x4
        tail = zr(5) + " - (" + "+".join(zr(m) for m in (1, 2, 3, 4)) + " + " + brk.join(
            fr(k, j) for k, j in eq.q) + ")"

    chunks = [P[n:n + _FACTORS_PER_LINE] for n in range(0, len(P), _FACTORS_PER_LINE)]
    lines = [sep.join(c) for c in chunks]
    body = (" " + brk + r" \\ &\quad " + brk + " ").join(lines)
    neg = tail.lstrip().startswith("-")              # хвост «- ζ…» без знака умножения перед ним
    if len(lines) > 1:
        body += (r" \\ &\quad " if neg else " " + brk + r" \\ &\quad " + brk + " ") + tail
    else:
        body += (" " if neg else (" \\cdot " if numeric else r"\,")) + tail

    if prefix is None:
        prefix = r"\frac{1}{\max X_{" + str(i) + r"}^{*}}"
    return (r"\begin{aligned}\frac{dX_{" + str(i) + r"}(t,p(t))}{dt} &= " + prefix
            + r"\Big( " + body + r" \Big)\end{aligned}")


def equation_conditions(i: int) -> List[str]:
    """Дополнительные строки под уравнением: f_k(X_j(t)) = 1 (если так сказано в документе) и пр."""
    out = []
    for r in f_table_rows():
        if r["i"] == i and (i, r["j"]) in FIXED_ONE:
            out.append(f"{_lx_f(r['f'], r['j'])} = 1")
    if i in DOC_NOTES:
        out.append(DOC_NOTES[i])
    return out


def equation_text(i: int) -> str:
    """Краткая текстовая запись уравнения (оставлена для совместимости)."""
    return equation_latex(i)
