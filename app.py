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
    widths = [max(0.45, 0.115 * len(plain(t["txt"]))) if "txt" in t else 1.25 for t in tokens]
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
    """Знаковая матрица X → X в виде обычной таблицы (как в Excel): значения 1, -1, 0, без графики."""
    cols = [f"X{n}" for n in range(1, M.N_X + 1)]
    head = "<th></th>" + "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(
        f"<tr><th>X{i + 1}</th>" + "".join(f"<td>{int(v)}</td>" for v in row) + "</tr>"
        for i, row in enumerate(S))
    css = ("<style>.mx{border-collapse:collapse;font-size:0.95rem}"
           ".mx th,.mx td{border:1px solid rgba(128,128,128,.7);padding:4px 12px;text-align:center}"
           ".mx td{font-style:italic}.mx th{font-weight:700}</style>")
    return css + f'<div style="overflow-x:auto"><table class="mx"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


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


def render_equations():
    """Уравнения в том виде, как записаны в документе: 1/max Xᵢ* в числителе, f(X(t)), ζ(t), p(t)."""
    st.write("Уравнения модели в записи документа. Масштаб τ в формулы не входит: он используется только "
             "в расчёте (множитель правой части), см. вкладку «Общие параметры».")
    for i in range(1, M.N_X + 1):
        st.markdown(f"Общий вид уравнения, характеризующего изменение во времени {M.DOC_DESCRIPTIONS[i]} "
                    f"(переменная X{sub(i)}):")
        st.latex(M.equation_latex(i) + r"\qquad (" + M.DOC_NUMBERS[i] + ")")

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
            f"- X(0) (начальные условия) — {R('x')};\n"
            f"- X* (рекомендованное значение) — {R('xstar')};\n"
            f"- γ (вес) — {R('gamma')}."
        )
        b1, b2, _ = st.columns([1, 1, 3])
        rnd_x0 = b1.button("Рандомизировать X(0)", key=k("rx0"))
        rnd_xstar = b2.button("Рандомизировать X*", key=k("rxs"))
        df_ind = pd.DataFrame({
            "№": range(1, M.N_X + 1),
            "Показатель": M.X_NAMES,
            "X(0)": [d["x0"] for d in base["indicators"]],
            "X* (рекоменд.)": [d["xstar"] for d in base["indicators"]],
            "Xmin": [d["xmin"] for d in base["indicators"]],
            "Xmax": [d["xmax"] for d in base["indicators"]],
            "γ (вес)": [d["gamma"] for d in base["indicators"]],
        })
        num = st.column_config.NumberColumn
        ed_ind = st.data_editor(
            df_ind, hide_index=True, use_container_width=True, key=k("ind"), num_rows="fixed",
            disabled=["№", "Показатель"],
            column_config={
                "X(0)": num(label=f"X(0)  [{R('x')}]", format="%.4g", required=True,
                            min_value=LIM["x"][0], max_value=LIM["x"][1],
                            help=f"Допустимо: {R('x')}"),
                "X* (рекоменд.)": num(label=f"X*  [{R('xstar')}]", format="%.4g", required=True,
                                      min_value=LIM["xstar"][0], max_value=LIM["xstar"][1],
                                      help=f"Допустимо: {R('xstar')}"),
                "Xmin": num(label=f"Xmin  [{R('x')}]", format="%.4g", required=True,
                            min_value=LIM["x"][0], max_value=LIM["x"][1],
                            help=f"Допустимо: {R('x')}, меньше Xmax"),
                "Xmax": num(label=f"Xmax  [{R('x')}]", format="%.4g", required=True,
                            min_value=LIM["x"][0], max_value=LIM["x"][1],
                            help=f"Допустимо: {R('x')}, больше Xmin"),
                "γ (вес)": num(label=f"γ (вес)  [{R('gamma')}]", format="%.4g", required=True,
                               min_value=LIM["gamma"][0], max_value=LIM["gamma"][1],
                               help=f"Допустимо: {R('gamma')}"),
            },
            height=35 * (M.N_X + 1) + 3,
        )
        for _, row in ed_ind.iterrows():
            n = int(row["№"]) - 1
            cfg["indicators"][n] = {
                "x0": float(row["X(0)"]), "xstar": float(row["X* (рекоменд.)"]),
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
            "Вспомогательные функции fₖ(Xⱼ) заданы полиномами **f(x) = a₀ + a₁·x + a₂·x² + a₃·x³**. "
            "В документе они названы «полиномами невысокой степени», коэффициенты не приведены — "
            "их можно менять ниже: у каждого уравнения есть строка «Задать для всех функций уравнения» "
            "(поля a₀…a₃ и кнопка), а в каждой строке функции поля стоят справа от обозначений. Редактируются все строки; "
            "по документу f₁(X₁), f₁₆(X₂), f₆₇(X₆), f₁₀₄(X₁₀) равны 1 (a₀ = 1, остальные 0)."
        )
        st.info(
            "**Допустимые значения:**\n"
            f"- коэффициенты a₀, a₁, a₂, a₃ (и в строке функции, и в поле «для всех функций уравнения») — {R('coef')};\n"
            f"- x (значение аргумента, при котором считается f) и Xⱼ(0) (начальное значение показателя) — {R('x')}."
        )
        rnd_f = st.button("Рандомизировать коэффициенты f", key=k("rf"))

        st.divider()
        s1, s2 = st.columns([3, 2])
        show_all = s2.toggle("Показать функции всех уравнений сразу", key="f_show_all",
                             help="Все 153 строки на одной странице. Полей много, поэтому страница может "
                                  "работать медленнее.")
        if show_all:
            eq_list = list(range(1, M.N_X + 1))
            s1.caption("Показаны все 14 уравнений.")
        else:
            eq_list = [s1.selectbox("Уравнение (показатель Xᵢ), функции которого редактируются",
                                    list(range(1, M.N_X + 1)),
                                    format_func=lambda i: f"X{sub(i)} — {M.X_NAMES[i - 1]}", key="feq")]
        st.caption("В строке: f(Xⱼ) и справа — её значение при введённом x; коэффициенты a₀…a₃; "
                   "x — значение аргумента (по умолчанию равно Xⱼ(0)); Xⱼ(0) — начальное значение показателя Xⱼ. "
                   "Изменение Xⱼ(0) меняет его и в таблице «Показатели», и во всех строках с этим аргументом.")
        for eq_i in eq_list:
            if show_all:
                st.markdown(f"**Уравнение X{sub(eq_i)} — {M.X_NAMES[eq_i - 1]}**")
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
            for r in [r for r in M.f_table_rows() if r["i"] == eq_i]:
                j, fk = r["j"], r["f"]
                key = f"{eq_i},{j}"
                cur = base["f_coefs"][key]
                x0cur = float(cfg["indicators"][j - 1]["x0"])
                kx = f"{x0cur:.6f}"        # значение X_j(0) в ключе: при его смене поля x и X_j(0) создаются заново
                xs = f"X{sub(j)}"

                def A(n, cur=cur, key=key, fk=fk):
                    return {"num": f"a{n}", "label": f"f{fk}: a{n}", "value": cur[n], "lim": LIM["coef"],
                            "key": k(f"fc{key}_{n}")}

                def f_value(out):
                    return f"{sum(out[f'a{n}'] * out['x'] ** n for n in range(4)):.4f}"

                out = formula_row([
                    {"txt": f"**f{sub(fk)}({xs})**"},
                    {"calc": f_value, "label": f"Значение f{fk} в уравнении X{eq_i}"},
                    {"txt": "= a₀"}, A(0),
                    {"txt": "+ a₁·x"}, A(1),
                    {"txt": "+ a₂·x²"}, A(2),
                    {"txt": "+ a₃·x³"}, A(3),
                    {"txt": "x ="}, {"num": "x", "label": f"f{fk}: x", "value": x0cur, "lim": LIM["x"],
                                     "key": k(f"fx{key}_{kx}")},
                    {"txt": f"{xs}(0) ="}, {"num": "x0", "label": f"f{fk}: {xs}(0)", "value": x0cur,
                                            "lim": LIM["x"], "key": k(f"fx0{key}_{kx}")},
                ])
                new = [float(out[f"a{n}"]) for n in range(4)]
                cfg["f_coefs"][key] = new
                base["f_coefs"][key] = list(new)     # запоминаем введённое при переключении между уравнениями
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
        st.caption("Строка — показатель Xᵢ, столбец — влияющий на него показатель Xⱼ. "
                   "1 — Xⱼ ускоряет рост Xᵢ, −1 — замедляет, 0 — не влияет. "
                   "Таблицу можно выделить и скопировать в Excel.")
        st.markdown(matrix_html(M.influence_sign_matrix(cfg)[:, :M.N_X]), unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
#  Формулы
# --------------------------------------------------------------------------- #
with tab_eq:
    render_equations()
