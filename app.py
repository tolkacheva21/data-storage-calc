"""
Интерфейс расчёта характеристик эффективности ПО хранилища данных.

Запуск:   streamlit run app.py
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import streamlit as st

import model as M
import plots

st.set_page_config(page_title="Моделирование характеристик хранилища данных", layout="wide")


def _rerun():
    (getattr(st, "rerun", None) or st.experimental_rerun)()


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

# --------------------------------------------------------------------------- #
#  Боковая панель: сохранение
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

cfg = copy.deepcopy(base)
apply_poly = False
rnd_x0 = rnd_xstar = rnd_zeta = rnd_f = False

with tab_in:
    sub_gen, sub_ind, sub_zeta, sub_f, sub_plan = st.tabs(
        ["Общие параметры", "Показатели Xᵢ", "Возмущения ζₘ(t)", "Функции fₖ", "План мероприятий"]
    )

    # ---- общие параметры
    with sub_gen:
        c1, c2, c3 = st.columns(3)
        cfg["T"] = c1.number_input("Горизонт моделирования T", min_value=0.1, value=float(base["T"]),
                                   step=0.1, key=k("T"),
                                   help="По заданию график строится на отрезке t от 0 до 1.")
        cfg["dt"] = c2.number_input("Шаг интегрирования Δt", min_value=0.001, value=float(base["dt"]),
                                    step=0.01, format="%.3f", key=k("dt"))
        cfg["tau"] = c3.number_input(
            "Временной масштаб τ", min_value=0.0001, value=float(base["tau"]), step=0.01, format="%.4f",
            key=k("tau"),
            help="Множитель правой части всех уравнений. В документе не задан: "
                 "определяет, насколько быстро показатели движутся к границам [Xmin, Xmax].",
        )
        st.caption("Метод интегрирования: Рунге–Кутта 4-го порядка; значения Xᵢ ограничиваются отрезком [Xᵢmin, Xᵢmax].")

    # ---- показатели
    with sub_ind:
        st.write("Начальные значения, рекомендованные значения Xᵢ*, границы и веса показателей (γᵢ).")
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
                "X(0)": num(format="%.4g"), "X* (рекоменд.)": num(format="%.4g", min_value=0.0001),
                "Xmin": num(format="%.4g"), "Xmax": num(format="%.4g"),
                "γ (вес)": num(format="%.4g", min_value=0.0),
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
        st.write("Профиль каждого возмущения ζₘ(t) (значения ограничиваются отрезком [0, 1]).")
        st.caption(
            "Постоянное: v0 · Ступенька: v0 до t0, затем v1 · Линейный: от v0 (в t0) до v1 (в t1) · "
            "Синусоида: v0 + v1·sin(2π(t−t0)/t1), t1 — период · Импульс: v1 на [t0, t1], иначе v0."
        )
        rnd_zeta = st.button("Рандомизировать возмущения", key=k("rz"))
        kinds = list(M.ZETA_KINDS)
        for m in range(M.N_Z):
            b = base["zeta"][m]
            st.markdown(f"**{M.ZETA_NAMES[m]}**")
            c0, c1, c2, c3, c4 = st.columns([2, 1, 1, 1, 1])
            kind = c0.selectbox("Вид", kinds, index=kinds.index(b["kind"]),
                                format_func=lambda s: M.ZETA_KINDS[s], key=k(f"zk{m}"))
            v0 = c1.number_input("v0", value=float(b["v0"]), step=0.05, key=k(f"zv0{m}"))
            v1 = c2.number_input("v1", value=float(b["v1"]), step=0.05, key=k(f"zv1{m}"))
            t0 = c3.number_input("t0", value=float(b["t0"]), step=0.1, key=k(f"zt0{m}"))
            t1 = c4.number_input("t1", value=float(b["t1"]), step=0.1, key=k(f"zt1{m}"))
            cfg["zeta"][m] = {"kind": kind, "v0": v0, "v1": v1, "t0": t0, "t1": t1}

    # ---- функции f
    with sub_f:
        st.write(
            "Вспомогательные функции fₖ(Xⱼ) заданы полиномами **f(x) = a0 + a1·x + a2·x² + a3·x³**. "
            "В документе они названы «полиномами невысокой степени», коэффициенты не приведены — "
            "их можно менять ниже. Функции f₁(X₁), f₁₆(X₂), f₆₇(X₆), f₁₀₄(X₁₀) по документу равны 1."
        )
        c1, c2, c3, c3b, c4 = st.columns([1, 1, 1, 1, 3])
        d0 = c1.number_input("a0", value=float(M.DEFAULT_POLY[0]), step=0.05, key=k("d0"))
        d1 = c2.number_input("a1", value=float(M.DEFAULT_POLY[1]), step=0.05, key=k("d1"))
        d2 = c3.number_input("a2", value=float(M.DEFAULT_POLY[2]), step=0.05, key=k("d2"))
        d3 = c3b.number_input("a3", value=float(M.DEFAULT_POLY[3]), step=0.05, key=k("d3"))
        c4.write("")
        apply_poly = c4.button("Применить a0…a3 ко всем функциям (кроме f ≡ 1)")
        rnd_f = c4.button("Рандомизировать коэффициенты f", key=k("rf"))

        rows = M.f_table_rows()
        df_f = pd.DataFrame({
            "Уравнение i": [r["i"] for r in rows],
            "Аргумент j": [r["j"] for r in rows],
            "Функция": [f"f{r['f']}(X{r['j']})" for r in rows],
            "a0": [base["f_coefs"][f"{r['i']},{r['j']}"][0] for r in rows],
            "a1": [base["f_coefs"][f"{r['i']},{r['j']}"][1] for r in rows],
            "a2": [base["f_coefs"][f"{r['i']},{r['j']}"][2] for r in rows],
            "a3": [base["f_coefs"][f"{r['i']},{r['j']}"][3] for r in rows],
        })
        ed_f = st.data_editor(
            df_f, hide_index=True, use_container_width=True, key=k("fed"), num_rows="fixed",
            disabled=["Уравнение i", "Аргумент j", "Функция"], height=420,
            column_config={c: st.column_config.NumberColumn(format="%.4g") for c in ("a0", "a1", "a2", "a3")},
        )
        for _, row in ed_f.iterrows():
            cfg["f_coefs"][f"{int(row['Уравнение i'])},{int(row['Аргумент j'])}"] = [
                float(row["a0"]), float(row["a1"]), float(row["a2"]), float(row["a3"])]

    # ---- план мероприятий
    with sub_plan:
        st.write(
            "План мероприятий p(t) задаётся как добавка uᵢ к скорости изменения показателя Xᵢ "
            "(в единицах показателя за единицу времени) начиная с момента t_старт. "
            "Нули — расчёт без мероприятий."
        )
        cfg["plan"]["start"] = st.number_input("Момент начала плана t_старт", min_value=0.0,
                                               value=float(base["plan"]["start"]), step=0.1, key=k("pstart"))
        df_u = pd.DataFrame({"№": range(1, M.N_X + 1), "Показатель": M.X_NAMES,
                             "uᵢ": [float(v) for v in base["plan"]["u"]]})
        ed_u = st.data_editor(df_u, hide_index=True, use_container_width=True, key=k("plan"), num_rows="fixed",
                              disabled=["№", "Показатель"], height=35 * (M.N_X + 1) + 3,
                              column_config={"uᵢ": st.column_config.NumberColumn(format="%.4g")})
        cfg["plan"]["u"] = [float(v) for v in ed_u["uᵢ"]]

# Применение стандартных коэффициентов ко всем функциям
if apply_poly:
    for key in cfg["f_coefs"]:
        i, j = (int(s) for s in key.split(","))
        if (i, j) not in M.FIXED_ONE:
            cfg["f_coefs"][key] = [d0, d1, d2, d3]
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
# --------------------------------------------------------------------------- #
problems = []
for n, d in enumerate(cfg["indicators"], start=1):
    if d["xmax"] <= d["xmin"]:
        problems.append(f"X{n}: Xmax должен быть больше Xmin")
if sum(d["gamma"] for d in cfg["indicators"]) <= 0:
    problems.append("Сумма весов γ должна быть положительной")
if problems:
    st.error("Проверьте входные данные:\n\n- " + "\n- ".join(problems))
    st.stop()

try:
    res = M.simulate(cfg)
except Exception as exc:  # noqa: BLE001
    st.error(f"Ошибка расчёта: {exc}")
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

    series = pd.DataFrame(res.X, columns=[f"X{i}" for i in range(1, M.N_X + 1)])
    series.insert(0, "t", res.t)
    for m in range(M.N_Z):
        series[f"zeta{m + 1}"] = res.zeta[:, m]
    series["Z"] = res.Z
    st.download_button("Скачать временные ряды (CSV)", series.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name="dwh_results.csv", mime="text/csv")

# --------------------------------------------------------------------------- #
#  Графики
# --------------------------------------------------------------------------- #
with tab_graph:
    g1, g2, g3, g4, g5, g6, g7 = st.tabs([
        "Динамика Xᵢ(t)", "Все показатели", "Возмущения ζ(t)", "Зависимость Xᵢ от Xⱼ",
        "Зависимость от ζ", "Лепестковая диаграмма", "Сечения по времени",
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

    with g4:
        c1, c2 = st.columns(2)
        yi = c1.selectbox("Показатель по оси Y (Xᵢ)", options, index=0, format_func=fmt, key="ph_i")
        xj = c2.selectbox("Показатель по оси X (Xⱼ)", options, index=1, format_func=fmt, key="ph_j")
        st.pyplot(plots.fig_phase(res, yi, xj))

    with g5:
        c1, c2 = st.columns([1, 2])
        zi = c1.selectbox("Возмущение", list(range(1, M.N_Z + 1)), format_func=lambda m: M.ZETA_NAMES[m - 1],
                          key="sens_z")
        sel_s = c2.multiselect("Показатели", options, default=[0, 1, 3, 4, 8, 13], format_func=fmt, key="sel_sens")
        st.caption("Возмущение ζₘ фиксируется на постоянном уровне 0…1, остальные берутся из входных данных; "
                   "показаны значения в конце горизонта T.")
        sens = M.sensitivity(cfg, zi, np.linspace(0.0, 1.0, 21))
        st.pyplot(plots.fig_sensitivity(sens, zi, sel_s))

    with g6:
        col_plot, col_ctrl = st.columns([1, 1])
        step = float(res.t[1] - res.t[0])
        with col_ctrl:
            tsel = st.slider("Момент времени t", min_value=0.0, max_value=float(res.t[-1]),
                             value=float(res.t[-1]), step=step, key="radar_t")
            show0 = st.checkbox("Начальные значения", value=True, key="radar_0")
            show_star = st.checkbox("Рекомендованные X*", value=True, key="radar_s")
        idx = int(np.argmin(np.abs(res.t - tsel)))
        with col_plot:
            st.pyplot(plots.fig_radar(res, idx, show0, show_star))

    with g7:
        st.caption("Лепестковые диаграммы в моменты t = 0, 0.2T, 0.4T, 0.6T, 0.8T, T "
                   "(при T = 1: 0; 0.2; 0.4; 0.6; 0.8; 1).")
        st.pyplot(plots.fig_radar_grid(res))

# --------------------------------------------------------------------------- #
#  Формулы
# --------------------------------------------------------------------------- #
with tab_eq:
    st.write("Структура уравнений, реализованная в коде (fₖ — полиномы a0 + a1·x + a2·x², τ — временной масштаб):")
    for i in range(1, M.N_X + 1):
        st.code(M.equation_text(i), language=None)
