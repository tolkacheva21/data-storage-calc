"""
Интерфейс расчёта характеристик эффективности ПО хранилища данных.

Запуск:   streamlit run app.py
"""
from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd
import streamlit as st

import model as M
import plots

st.set_page_config(page_title="Моделирование характеристик хранилища данных", layout="wide")

LIM = M.LIMITS

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

COL_X0 = "Начальные условия X(0)"
COL_XSTAR = "Калибровочный коэффициент X*"


def sub(n) -> str:
    """Число нижним индексом: 12 -> ₁₂."""
    return str(n).translate(_SUB)


def R(key: str) -> str:
    """Допустимый диапазон в виде «0.1…10» - для подписей полей."""
    lo, hi = LIM[key]
    return f"{lo:g}…{hi:g}"


def _rerun():
    (getattr(st, "rerun", None) or st.experimental_rerun)()


def _box():
    """Рамка вокруг блока (на старых версиях Streamlit - без рамки)."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def _cols(widths):
    """Колонки с выравниванием по центру (на старых версиях Streamlit - без выравнивания)."""
    try:
        return st.columns(widths, vertical_alignment="center")
    except TypeError:
        return st.columns(widths)


def formula_row(tokens: list) -> dict:
    """Строка-формула: текст и поля ввода в одну линию, поле стоит сразу после своего обозначения.

    Элементы tokens:
      {"txt": "..."}                        - текст (markdown);
      {"num": имя, "label", "value", "lim": (min, max), "key", "step", "disabled"} - поле ввода числа;
      {"btn": имя, "label", "key"}          - кнопка (в результате True, если нажата);
      {"calc": функция(значения) -> str, "label"} - вычисляемое поле (только чтение);
      {"show": "текст", "label"}            - поле только для чтения.
    Возвращает словарь {имя: введённое число}."""
    plain = lambda s: re.sub(r"<[^>]+>|\*\*", "", s)   # noqa: E731  длина текста без разметки
    widths = [t["w"] if "w" in t else (max(0.45, 0.115 * len(plain(t["txt"]))) if "txt" in t else 1.25)
              for t in tokens]
    cols = _cols(widths)
    out, deferred = {}, []
    for col, t in zip(cols, tokens):
        with col:
            if "txt" in t:
                st.markdown(t["txt"], unsafe_allow_html=True)
            elif "btn" in t:
                out[t["btn"]] = st.button(t["label"], key=t["key"])
            elif "num" in t:
                lo, hi = t["lim"]
                out[t["num"]] = st.number_input(
                    t["label"], min_value=float(lo), max_value=float(hi), value=float(t["value"]),
                    step=t.get("step", 0.05), format="%.3f", key=t["key"],
                    disabled=t.get("disabled", False), label_visibility="collapsed")
            else:
                deferred.append((col, t))
    for col, t in deferred:          # вычисляемые поля заполняем после того, как введённые значения известны
        with col:
            text = t["calc"](out) if "calc" in t else t["show"]
            st.text_input(t["label"], value=text, disabled=True, label_visibility="collapsed")
    return out


def matrix_html(S) -> str:
    """Знаковая матрица (строки - X1..X14, колонки - X1..X14 и ζ1..ζ5) в виде обычной таблицы:
    значения 1, -1, 0, без графики."""
    xcols = [f"X{n}" for n in range(1, M.N_X + 1)]
    zcols = [f"ζ{m}" for m in range(1, M.N_Z + 1)]
    head1 = (f'<tr><th rowspan="2">Показатель</th>'
             f'<th colspan="{M.N_X}">Влияние показателей Xⱼ</th>'
             f'<th colspan="{M.N_Z}" class="zs">Влияние возмущений ζₘ</th></tr>')
    head2 = ("<tr>"
             + "".join(f'<th title="{M.X_NAMES[n]}">{c}</th>' for n, c in enumerate(xcols))
             + "".join(f'<th class="{"zs" if m == 0 else ""}" title="{M.ZETA_NAMES[m]}">{c}</th>'
                       for m, c in enumerate(zcols))
             + "</tr>")
    body = ""
    for i, row in enumerate(S):
        cells = "".join(f'<td class="{"zs" if c == M.N_X else ""}">{int(v)}</td>' for c, v in enumerate(row))
        body += f'<tr><th class="rl" title="{M.X_NAMES[i]}">{M.X_SHORT[i]}</th>{cells}</tr>'
    css = ("<style>.mx{border-collapse:collapse;font-size:0.95rem}"
           ".mx th,.mx td{border:1px solid rgba(128,128,128,.7);padding:4px 12px;text-align:center}"
           ".mx td{font-style:italic}.mx th{font-weight:700}"
           ".mx .rl{text-align:left;white-space:nowrap}"
           ".mx .zs{border-left:3px solid rgba(128,128,128,1)}</style>")
    return (css + '<div style="overflow-x:auto"><table class="mx"><thead>' + head1 + head2
            + f"</thead><tbody>{body}</tbody></table></div>")


# --------------------------------------------------------------------------- #
#  Состояние: базовая конфигурация (значения по умолчанию для виджетов)
# --------------------------------------------------------------------------- #
if "base_cfg" not in st.session_state:
    st.session_state.base_cfg = M.default_config()
    st.session_state.ver = 0

base = st.session_state.base_cfg
ver = st.session_state.ver


def k(name: str) -> str:
    """Ключ виджета; при смене ver (загрузка/сброс) все поля пересоздаются."""
    return f"{name}_{ver}"


st.title("Моделирование характеристик эффективности ПО хранилища данных")
st.caption(
    "Модель системной динамики: 14 показателей Xᵢ(t), 5 возмущений ζₘ(t)."
)
err_slot = st.empty()   # сюда выводятся ошибки ввода (заполняется после проверки данных)

# --------------------------------------------------------------------------- #
#  Боковая панель: генерация данных
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("Генерация данных")
    rnd_all = st.button("Рандомизировать всё", help="X(0), X*, возмущения ζ и коэффициенты функций f")
    if st.button("Сбросить все значения по умолчанию"):
        st.session_state.base_cfg = M.default_config()
        st.session_state.ver += 1
        _rerun()

# --------------------------------------------------------------------------- #
#  Ввод данных
# --------------------------------------------------------------------------- #
tab_in, tab_res, tab_graph, tab_eq = st.tabs(
    ["1. Ввод данных", "2. Результаты расчёта", "3. Графики и диаграммы", "Формулы модели"]
)


def _poly_latex(c, x, j) -> str:
    """Полином с подставленными коэффициентами (из вкладки «Функции fₖ»), затем с подставленным
    значением X_j:  a0 + a1·X_j + ...  =  a0 + a1·0.800 + ..."""
    terms_sym, terms_num = [], []
    for n in range(4):
        if n == 0:
            terms_sym.append(f"{c[0]:g}")
            terms_num.append(f"{c[0]:g}")
        elif abs(c[n]) > 1e-12:
            pw = "" if n == 1 else f"^{{{n}}}"
            terms_sym.append(f"{c[n]:g}\\cdot X_{{{j}}}{pw}")
            terms_num.append(f"{c[n]:g}\\cdot{x:.3f}{pw}")
    return " + ".join(terms_sym) + " = " + " + ".join(terms_num)


def _prod_latex(name: str, vals) -> str:
    """P = 1.0000 · 1.0600 · … = 0.7312 (длинное произведение переносится по 7 множителей)."""
    chunks = [vals[n:n + 7] for n in range(0, len(vals), 7)]
    lines = [" \\cdot ".join(f"{v:.4f}" for v in ch) for ch in chunks]
    body = " \\cdot \\\\ &\\quad ".join(lines)
    return f"\\begin{{aligned}}{name} &= {body} = {float(np.prod(vals)):.4f}\\end{{aligned}}"


def render_substitution(cfg: dict, res, idx: int, i: int):
    """Под уравнением X_i: как в него подставляются f_k, ζ_m и что получается для dX_i/dt в момент t."""
    eq = M.EQUATIONS[i]
    x, z = res.X[idx], res.zeta[idx]
    tau = float(cfg["tau"])
    xstar = float(cfg["indicators"][i - 1]["xstar"])

    def f_lines(items):
        lines, vals = [], []
        for kf, j in items:
            c = [float(v) for v in cfg["f_coefs"][f"{i},{j}"]] + [0.0] * 4
            xv = float(x[j - 1])
            val = M.f_value(cfg, i, j, xv)
            vals.append(val)
            lines.append(f"f_{{{kf}}}(X_{{{j}}}) &= {_poly_latex(c, xv, j)} = {val:.4f}")
        return lines, vals

    def show_lines(lines):
        if len(lines) > 4:                       # длинные списки - в две колонки
            half = (len(lines) + 1) // 2
            for col, part in zip(st.columns(2), (lines[:half], lines[half:])):
                col.latex(r"\begin{aligned}" + r" \\ ".join(part) + r"\end{aligned}")
        else:
            st.latex(r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}")

    p_lines, p_vals = f_lines(eq.p)
    q_lines, q_vals = f_lines(eq.q)
    P = float(np.prod(p_vals)) if p_vals else 1.0
    Q = float(np.prod(q_vals)) if q_vals else 1.0

    zs = lambda ms: "(" + " + ".join(f"{z[m - 1]:.4f}" for m in ms) + ")"   # noqa: E731
    qs = f"{Q:.4f}\\cdot" if q_vals else ""
    if eq.kind == "std":
        v = P * sum(z[m - 1] for m in eq.plus) - Q * sum(z[m - 1] for m in eq.minus)
        expr = f"{P:.4f}\\cdot{zs(eq.plus)}"
        if eq.minus:
            expr += f" - {qs}{zs(eq.minus)}"
    elif eq.kind == "sub":
        v = P - Q * sum(z[m - 1] for m in eq.minus)
        expr = f"{P:.4f} - {qs}{zs(eq.minus)}"
    elif eq.kind == "x4":
        v = P * z[4] - (sum(z[m - 1] for m in (1, 2, 3, 4)) + Q)
        expr = f"{P:.4f}\\cdot{z[4]:.4f} - ({zs((1, 2, 3, 4))} + {Q:.4f})"
    else:
        return
    dx = tau * v / xstar

    st.markdown(f"**Подстановка чисел в уравнение X{sub(i)} при t = {res.t[idx]:.2f}** "
                f"(значения Xⱼ(t) и ζₘ(t) взяты из расчёта):")
    st.markdown("1) Значения функций fₖ(Xⱼ): коэффициенты a₀…a₃ из вкладки «Функции fₖ», "
                "затем вместо Xⱼ подставлено его значение в момент t:")
    show_lines(p_lines)
    st.markdown("2) Произведение функций" + (" (множитель P):" if q_vals else ":"))
    st.latex(_prod_latex("P", p_vals))
    if q_vals:
        st.markdown("Функции второго произведения (множитель Q):")
        show_lines(q_lines)
        st.latex(_prod_latex("Q", q_vals))
    st.markdown("3) Возмущения в этот момент и значение выражения в скобках v:")
    st.latex(r",\ ".join(f"\\zeta_{{{m}}}={z[m - 1]:.4f}" for m in range(1, M.N_Z + 1)))
    st.latex(f"v = {expr} = {v:.4f}")
    st.markdown("4) Скорость изменения показателя (с масштабом τ и множителем 1/X*):")
    st.latex(f"\\frac{{dX_{{{i}}}}}{{dt}} = \\tau\\cdot\\frac{{1}}{{X_{{{i}}}^{{*}}}}\\cdot v = "
             f"{tau:g}\\cdot\\frac{{1}}{{{xstar:g}}}\\cdot({v:.4f}) = {dx:.5f}")


def render_equations(cfg=None, res=None):
    """Уравнения в записи документа; под каждым - то же уравнение с числами и (по желанию)
    пошаговая подстановка. Числа берутся из вкладки «Ввод данных» (коэффициенты fₖ, возмущения,
    X*, τ) и из расчёта (значения Xⱼ(t) в выбранный момент)."""
    st.write("Уравнения модели в записи документа. Масштаб τ в символьные формулы не входит: он используется "
             "только в расчёте (множитель правой части), см. вкладку «Общие параметры».")
    idx = 0
    detail = True
    if res is not None:
        step = float(res.t[1] - res.t[0])
        tsel = st.slider("Момент времени t, для которого показана подстановка чисел под формулами",
                         min_value=0.0, max_value=float(res.t[-1]), value=0.0, step=step,
                         key=f"eq_t_{cfg['T']:g}_{cfg['dt']:g}")
        idx = int(np.argmin(np.abs(res.t - tsel)))
        detail = st.checkbox("Показывать пошаговую подстановку под каждой формулой", value=True,
                             key="eq_detail")
    else:
        st.info("Подстановка чисел под формулами появится после исправления ошибок на вкладке «1. Ввод данных».")
    for i in range(1, M.N_X + 1):
        st.markdown(f"Общий вид уравнения, характеризующего изменение во времени {M.DOC_DESCRIPTIONS[i]} "
                    f"(переменная X{sub(i)}):")
        st.latex(M.equation_latex(i) + r"\qquad (" + M.DOC_NUMBERS[i] + ")")
        if res is not None:
            x, z = res.X[idx], res.zeta[idx]
            tau = float(cfg["tau"])
            xstar = float(cfg["indicators"][i - 1]["xstar"])
            st.markdown(f"То же уравнение с числами при t = {res.t[idx]:.2f}: значения fₖ посчитаны по "
                        f"коэффициентам из «Функции fₖ», ζₘ — по профилям возмущений:")
            st.latex(M.equation_latex(
                i,
                f_repr=lambda kf, j, i=i: f"{M.f_value(cfg, i, j, float(x[j - 1])):.4f}",
                z_repr=lambda m: f"{z[m - 1]:.4f}",
                prefix=rf"{tau:g}\cdot\frac{{1}}{{{xstar:g}}}",
            ))
            if detail:
                render_substitution(cfg, res, idx, i)
            st.divider()

cfg = copy.deepcopy(base)
apply_eq = None      # (номер уравнения, [a0, a1, a2, a3]) - «задать для всех функций уравнения»
rnd_x0 = rnd_xstar = rnd_zeta = rnd_f = False
x0_changes = {}      # {номер показателя j: новое X_j(0)}, введённые в строках функций

with tab_in:
    sub_gen, sub_ind, sub_zeta, sub_f = st.tabs(
        ["Общие параметры", "Показатели Xᵢ", "Возмущения ζₘ(t)", "Функции fₖ"]
    )

    # ---- общие параметры
    with sub_gen:
        st.info(
            f"**Допустимые значения:** T (время t изменяется от 0 до T) — {R('T')}; "
            f"Δt — {R('dt')}; τ — {R('tau')}."
        )
        c1, c2, c3 = st.columns(3)
        cfg["T"] = c1.number_input(f"Горизонт T: время t от 0 до T  [{R('T')}]",
                                   min_value=LIM["T"][0], max_value=LIM["T"][1], value=float(base["T"]),
                                   step=0.1, key=k("T"),
                                   help=f"Допустимо: {R('T')}. По заданию график строится на отрезке t от 0 до 1.")
        cfg["dt"] = c2.number_input(f"Шаг интегрирования Δt  [{R('dt')}]",
                                    min_value=LIM["dt"][0], max_value=LIM["dt"][1], value=float(base["dt"]),
                                    step=0.01, format="%.3f", key=k("dt"),
                                    help=f"Допустимо: {R('dt')}. Если шаг слишком крупный, расчёт сам уточняет его "
                                         f"(не менее {M.MIN_STEPS} шагов на отрезке 0…T).")
        cfg["tau"] = c3.number_input(
            f"Временной масштаб τ  [{R('tau')}]", min_value=LIM["tau"][0], max_value=LIM["tau"][1],
            value=float(base["tau"]), step=0.01, format="%.4f", key=k("tau"),
            help=f"Допустимо: {R('tau')}. Множитель правой части всех уравнений. В документе не задан: "
                 "определяет, насколько быстро показатели движутся к границам [Xmin, Xmax].",
        )
        st.caption("Метод интегрирования: Рунге–Кутта 4-го порядка; значения Xᵢ ограничиваются отрезком [Xᵢmin, Xᵢmax].")

    # ---- показатели
    with sub_ind:
        st.write("Начальные значения, рекомендованные значения Xᵢ*, границы и веса показателей (γᵢ).")
        st.info(
            "**Допустимые значения:**\n"
            f"- Xmin и Xmax(минимум и максимум) — {R('x')}, при этом Xmax должен быть больше Xmin;\n"
            f"- Начальные условия X(0) — {R('x')};\n"
            f"- Калибровочный коэффициент X* — {R('xstar')};\n"
            f"- γ (вес) — {R('gamma')}."
        )
        b1, b2, _ = st.columns([1, 1, 3])
        rnd_x0 = b1.button("Рандомизировать начальные условия", key=k("rx0"))
        rnd_xstar = b2.button("Рандомизировать калибровочные коэффициенты", key=k("rxs"))
        df_ind = pd.DataFrame({
            "№": range(1, M.N_X + 1),
            "Показатель": M.X_NAMES,
            COL_X0: [d["x0"] for d in base["indicators"]],
            COL_XSTAR: [d["xstar"] for d in base["indicators"]],
            "Xmin": [d["xmin"] for d in base["indicators"]],
            "Xmax": [d["xmax"] for d in base["indicators"]],
            "γ (вес)": [d["gamma"] for d in base["indicators"]],
        })
        num = st.column_config.NumberColumn
        ed_ind = st.data_editor(
            df_ind, hide_index=True, use_container_width=True, key=k("ind"), num_rows="fixed",
            disabled=["№", "Показатель"],
            column_config={
                COL_X0: num(format="%.4g", required=True,
                            min_value=LIM["x"][0], max_value=LIM["x"][1]),
                COL_XSTAR: num(format="%.4g", required=True,
                               min_value=LIM["xstar"][0], max_value=LIM["xstar"][1]),
                "Xmin": num(format="%.4g", required=True,
                            min_value=LIM["x"][0], max_value=LIM["x"][1]),
                "Xmax": num(format="%.4g", required=True,
                            min_value=LIM["x"][0], max_value=LIM["x"][1]),
                "γ (вес)": num(format="%.4g", required=True,
                               min_value=LIM["gamma"][0], max_value=LIM["gamma"][1]),
            },
            height=35 * (M.N_X + 1) + 3,
        )
        for _, row in ed_ind.iterrows():
            n = int(row["№"]) - 1
            cfg["indicators"][n] = {
                "x0": float(row[COL_X0]), "xstar": float(row[COL_XSTAR]),
                "xmin": float(row["Xmin"]), "xmax": float(row["Xmax"]), "gamma": float(row["γ (вес)"]),
            }
    # ---- возмущения
    with sub_zeta:
        st.write("Возмущение ζₘ(t) — внешний фактор со значением от 0 до 1, которое меняется со временем. "
                 "Для каждого выберите вид зависимости: под выбором записана формула, по которой считается ζₘ(t), "
                 "а ниже — поля для её параметров (справа от каждого обозначения). "
                 "Результат расчёта всегда ограничивается отрезком [0; 1].")
        st.info(
            "**Допустимые значения:**\n"
            f"- v₀, v₁ (значения возмущения) — {R('zeta')};\n"
            f"- t₀, t₁ (моменты времени) — {R('t')}."
        )
        rnd_zeta = st.button("Рандомизировать возмущения", key=k("rz"))
        kinds = list(M.ZETA_KINDS)
        for m in range(M.N_Z):
            b = base["zeta"][m]
            z = f"ζ{sub(m + 1)}(t)"
            with _box():
                h1, h2 = st.columns([3, 2])
                h1.markdown(f"##### ζ{sub(m + 1)} — {M.ZETA_NAMES[m].split(' ', 1)[1]}")
                kind = h2.selectbox("Вид зависимости", kinds, index=kinds.index(b["kind"]),
                                    format_func=lambda s: M.ZETA_KINDS[s], key=k(f"zk{m}"))

                def P(name, step=0.05):
                    """Поле параметра возмущения m (внутреннее имя v0/v1/t0/t1)."""
                    return {"num": name, "label": f"{z}: {name}", "value": b[name], "step": step,
                            "lim": LIM["zeta"] if name.startswith("v") else LIM["t"], "key": k(f"z{name}{m}")}

                zl = f"\\zeta_{{{m + 1}}}(t)"
                if kind == "const":
                    formula = zl + r" = v_0"
                    tokens = [{"txt": "**v₀** (значение) ="}, P("v0")]
                elif kind == "step":
                    formula = zl + r" = \begin{cases} v_0, & t < t_0 \\ v_1, & t \ge t_0 \end{cases}"
                    tokens = [{"txt": "**v₀** (до скачка) ="}, P("v0"), {"txt": "**v₁** (после скачка) ="}, P("v1"),
                              {"txt": "**t₀** (момент скачка) ="}, P("t0", 0.1)]
                elif kind == "ramp":
                    formula = (zl + r" = \begin{cases} v_0, & t < t_0 \\ "
                               r"v_0 + (v_1 - v_0)\,\dfrac{t - t_0}{t_1 - t_0}, & t_0 \le t \le t_1 \\ "
                               r"v_1, & t > t_1 \end{cases}")
                    tokens = [{"txt": "**v₀** (значение в начале) ="}, P("v0"),
                              {"txt": "**v₁** (значение в конце) ="}, P("v1"),
                              {"txt": "**t₀** (начало изменения) ="}, P("t0", 0.1),
                              {"txt": "**t₁** (конец изменения) ="}, P("t1", 0.1)]
                elif kind == "sin":
                    formula = zl + r" = v_0 + v_1 \sin\dfrac{2\pi\,(t - t_0)}{t_1}"
                    tokens = [{"txt": "**v₀** (среднее значение) ="}, P("v0"),
                              {"txt": "**v₁** (амплитуда) ="}, P("v1"),
                              {"txt": "**t₀** (сдвиг по времени) ="}, P("t0", 0.1),
                              {"txt": "**t₁** (период) ="}, P("t1", 0.1)]
                else:  # pulse
                    formula = (zl + r" = \begin{cases} v_1, & t_0 \le t \le t_1 \\ "
                               r"v_0, & \text{в остальное время} \end{cases}")
                    tokens = [{"txt": "**v₀** (вне импульса) ="}, P("v0"),
                              {"txt": "**v₁** (в импульсе) ="}, P("v1"),
                              {"txt": "**t₀** (начало импульса) ="}, P("t0", 0.1),
                              {"txt": "**t₁** (конец импульса) ="}, P("t1", 0.1)]
                st.latex(formula)
                vals = {n: float(b[n]) for n in ("v0", "v1", "t0", "t1")}   # скрытые параметры вида сохраняются
                vals.update(formula_row(tokens))
            cfg["zeta"][m] = {"kind": kind, **vals}
            b.update(kind=kind, **vals)      # запоминаем введённое, чтобы оно не терялось при смене вида

    # ---- функции f
    with sub_f:
        st.write(
            "Вспомогательные функции fₖ(Xⱼ) заданы полиномами "
            "**fₖ(Xⱼ) = a₀ + a₁·Xⱼ + a₂·Xⱼ² + a₃·Xⱼ³**, "
            "где Xⱼ — показатель, который стоит в скобках функции в уравнении (например, в уравнении X₁ "
            "функция f₂ зависит от X₂, f₃ — от X₃ и т.д.; так же, как на вкладке «Формулы модели»). "
            "В документе они названы «полиномами невысокой степени», коэффициенты не приведены — "
            "их можно менять ниже. Функции сгруппированы по уравнениям: сначала все функции уравнения X₁ "
            "(f₁…f₁₄), ниже — уравнения X₂ (f₁₅…f₂₈) и т.д. У каждого уравнения есть строка "
            "«Для всех f уравнения» (поля a₀…a₃ и кнопка «Применить»). "
            "По документу f₁(X₁), f₁₆(X₂), f₆₇(X₆), f₁₀₄(X₁₀) равны 1 (a₀ = 1, остальные 0)."
        )
        st.info(
            "**Допустимые значения:**\n"
            f"- коэффициенты a₀, a₁, a₂, a₃ (и в строке функции, и в строке «для всех f уравнения») — {R('coef')};\n"
            f"- Xⱼ(0) (начальное значение показателя) — {R('x')}."
        )
        rnd_f = st.button("Рандомизировать коэффициенты f", key=k("rf"))
        st.caption("В строке: fₖ(Xⱼ), справа — её значение при текущем Xⱼ(0); коэффициенты a₀…a₃ стоят "
                   "рядом со своими степенями Xⱼ; Xⱼ(0) — начальное значение показателя Xⱼ. "
                   "Изменение Xⱼ(0) меняет его и в таблице «Показатели», и во всех строках, где используется Xⱼ.")
        st.divider()

        all_rows = M.f_table_rows()
        for eq_i in range(1, M.N_X + 1):
            rows = [r for r in all_rows if r["i"] == eq_i]
            fmin, fmax = min(r["f"] for r in rows), max(r["f"] for r in rows)
            st.markdown(f"#### Уравнение X{sub(eq_i)} — {M.X_NAMES[eq_i - 1]}  (f{sub(fmin)}…f{sub(fmax)})")
            with _box():
                eq_vals = formula_row(
                    [{"txt": f"**Для всех f уравнения X{sub(eq_i)}:**"}]
                    + [tk for n in range(4) for tk in (
                        {"txt": f"a{sub(n)} ="},
                        {"num": f"a{n}", "label": f"X{eq_i}, все f: a{n}", "value": M.DEFAULT_POLY[n],
                         "lim": LIM["coef"], "key": k(f"da{eq_i}_{n}")})]
                    + [{"btn": "apply", "label": "Применить", "key": k(f"dab{eq_i}")}])
            if eq_vals["apply"]:
                apply_eq = (eq_i, [float(eq_vals[f"a{n}"]) for n in range(4)])
            for r in rows:
                j, fk = r["j"], r["f"]
                key = f"{eq_i},{j}"
                cur = base["f_coefs"][key]
                xs = f"X{sub(j)}"                                   # аргумент функции - показатель Xⱼ
                x0cur = float(cfg["indicators"][j - 1]["x0"])      # его начальное значение Xⱼ(0)
                kx = f"{x0cur:.6f}"                                 # при смене Xⱼ(0) поля создаются заново

                def A(n, cur=cur, key=key, fk=fk):
                    return {"num": f"a{n}", "label": f"f{fk}: a{n}", "value": cur[n], "lim": LIM["coef"],
                            "key": k(f"fc{key}_{n}")}

                def f_value(out):
                    return f"{sum(out[f'a{n}'] * out['x0'] ** n for n in range(4)):.4f}"

                # "\\+" - чтобы markdown не превращал строку, начинающуюся с «+», в маркер списка
                out = formula_row([
                    {"txt": f"**f{sub(fk)}({xs})**", "w": 1.1},
                    {"calc": f_value, "label": f"Значение f{fk}(X{j}) в уравнении X{eq_i}"},
                    {"txt": " **=**", "w": 0.2},
                    A(0),
                    {"txt": " **+**", "w": 0.2},
                    A(1),
                    {"txt": f"·{xs} · ", "w": 0.4},

                    {"txt": " **+**", "w": 0.4},
                    A(2),
                    {"txt": f"·{xs}²   ", "w": 0.4},

                    {"txt": " **+**", "w": 0.2},
                    A(3),
                    {"txt": f"·{xs}³", "w": 0.4},
                    {"txt": f"{xs}(0) =", "w": 0.9},
                    {"num": "x0", "label": f"f{fk}: {xs}(0)", "value": x0cur, "lim": LIM["x"],
                     "key": k(f"fx0{key}_{kx}")},
                ])
                new = [float(out[f"a{n}"]) for n in range(4)]
                cfg["f_coefs"][key] = new
                base["f_coefs"][key] = list(new)     # запоминаем введённое
                if abs(float(out["x0"]) - x0cur) > 1e-12:
                    x0_changes[j] = float(out["x0"])

# Начальное значение X_j(0), введённое в строке функции, переносим в показатели и обновляем все поля
if x0_changes:
    for j, v in x0_changes.items():
        cfg["indicators"][j - 1]["x0"] = v
    st.session_state.base_cfg = cfg
    st.session_state.ver += 1
    _rerun()

# Коэффициенты a0…a3, заданные для всех функций одного уравнения
if apply_eq:
    ei, coefs = apply_eq
    for r in M.f_table_rows():
        if r["i"] == ei:
            cfg["f_coefs"][f"{ei},{r['j']}"] = list(coefs)
    st.session_state.base_cfg = cfg
    st.session_state.ver += 1
    _rerun()

# Генерация случайных данных: меняем только выбранные блоки, остальное сохраняется
if rnd_all or rnd_x0 or rnd_xstar or rnd_zeta or rnd_f:
    rng = np.random.default_rng()
    if rnd_all or rnd_x0:
        M.randomize_x0(cfg, rng)
    if rnd_all or rnd_xstar:
        M.randomize_xstar(cfg, rng)
    if rnd_all or rnd_zeta:
        M.randomize_zeta(cfg, rng)
    if rnd_all or rnd_f:
        M.randomize_f(cfg, rng)
    st.session_state.base_cfg = cfg
    st.session_state.ver += 1
    _rerun()

# --------------------------------------------------------------------------- #
#  Проверка данных и расчёт
#  validate_config - та же проверка, что выполняется внутри M.simulate(),
#  поэтому некорректные данные не пройдут и в обход интерфейса.
# --------------------------------------------------------------------------- #
problems = M.validate_config(cfg)
res = None
if not problems:
    try:
        res = M.simulate(cfg)
    except M.ConfigError as exc:
        problems = exc.problems
    except Exception as exc:  # noqa: BLE001
        problems = [f"Ошибка расчёта: {exc}"]

if problems:
    err_slot.error("**Расчёт не выполнен — исправьте входные данные:**\n\n"
                   + "\n".join(f"- {p}" for p in problems))
    with tab_res:
        st.info("Результаты появятся после исправления ошибок на вкладке «1. Ввод данных».")
    with tab_graph:
        st.info("Графики появятся после исправления ошибок на вкладке «1. Ввод данных».")
    with tab_eq:
        render_equations()
    st.stop()

# --------------------------------------------------------------------------- #
#  Результаты
# --------------------------------------------------------------------------- #
with tab_res:
    m1, m2, m3 = st.columns(3)
    m1.metric("Целевая функция Z(T), в конце", f"{res.Z_final:.5f}")
    m2.metric("Целевая функция Z, среднее по времени", f"{res.Z_mean:.5f}")
    m3.metric("Z(0), в начале", f"{res.Z[0]:.5f}")
    st.caption("Z = Σ γᵢ·((Xᵢ − Xᵢ*)/(Xᵢmax − Xᵢmin))², веса γᵢ нормируются на единицу. Чем меньше Z, тем ближе "
               "состояние хранилища к рекомендованному.")

    ind = cfg["indicators"]
    table = pd.DataFrame({
        "Показатель": M.X_NAMES,
        "X(0)": res.X[0],
        "X(T)": res.X[-1],
        "X* (рекоменд.)": [d["xstar"] for d in ind],
        "X(T) − X*": res.X[-1] - np.array([d["xstar"] for d in ind]),
        "Xmin": [d["xmin"] for d in ind],
        "Xmax": [d["xmax"] for d in ind],
        "max за период": res.X.max(axis=0),
        "min за период": res.X.min(axis=0),
    }, index=[f"X{i}" for i in range(1, M.N_X + 1)])
    st.dataframe(table.style.format("{:.4f}", subset=table.columns[1:]), use_container_width=True,
                 height=35 * (M.N_X + 1) + 3)

# --------------------------------------------------------------------------- #
#  Графики
# --------------------------------------------------------------------------- #
with tab_graph:
    g1, g2, g3, g6, g7, g8 = st.tabs([
        "Динамика Xᵢ(t)", "Все показатели", "Возмущения ζ(t)",
        "Лепестковая диаграмма", "Сечения по времени", "Матрица взаимовлияния",
    ])
    options = list(range(M.N_X))
    fmt = lambda i: M.X_SHORT[i]  # noqa: E731

    with g1:
        sel = st.multiselect("Показатели", options, default=options[:6], format_func=fmt, key="sel_dyn")
        st.pyplot(plots.fig_dynamics(res, sel))

    with g2:
        st.pyplot(plots.fig_small_multiples(res))

    with g3:
        st.pyplot(plots.fig_zeta(res))

    with g6:
        col_plot, col_ctrl = st.columns([1, 1])
        step = float(res.t[1] - res.t[0])
        with col_ctrl:
            tsel = st.slider("Момент времени t", min_value=0.0, max_value=float(res.t[-1]),
                             value=float(res.t[-1]), step=step,
                             key=f"radar_t_{cfg['T']:g}_{cfg['dt']:g}")
            show0 = st.checkbox("Начальные значения", value=True, key="radar_0")
            show_star = st.checkbox("Рекомендованные X*", value=True, key="radar_s")
        idx = int(np.argmin(np.abs(res.t - tsel)))
        with col_plot:
            st.pyplot(plots.fig_radar(res, idx, show0, show_star))

    with g7:
        st.caption("Лепестковые диаграммы в моменты t = 0, 0.2T, 0.4T, 0.6T, 0.8T, T "
                   "(при T = 1: 0; 0.2; 0.4; 0.6; 0.8; 1).")
        st.pyplot(plots.fig_radar_grid(res))

    with g8:
        st.caption("Строка — показатель Xᵢ, столбец — влияющий на него показатель Xⱼ или возмущение ζₘ. "
                   "1 — фактор ускоряет рост Xᵢ, −1 — замедляет, 0 — не влияет. "
                   "Таблицу можно выделить и скопировать в Excel.")
        st.markdown(matrix_html(M.influence_sign_matrix(cfg)), unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
#  Формулы
# --------------------------------------------------------------------------- #
with tab_eq:
    render_equations(cfg, res)