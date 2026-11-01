"""
Figuras de las pestañas Modelos y Predicción.

Van aparte de utils/figures.py porque aquel módulo describe los datos y este
describe modelos: las dos familias no comparten ni fuentes ni criterios.

Color, con las mismas reglas del resto del tablero (ver utils/theme.py):
  * Azul y coral siguen significando LGG y GBM, y solo eso. Aquí aparecen cuando
    la marca habla de un grado: la probabilidad que asigna un modelo, el sentido
    en que una variable empuja una predicción, los pacientes reales de cada
    grado.
  * Los tres modelos NO son grados, así que no se pintan con azul ni coral. Se
    distinguen con tres tintas neutras de distinta claridad, un trazo distinto
    (continuo, discontinuo, punteado) y siempre con su nombre en la leyenda o
    junto a la línea: el color nunca es la única pista.
  * Magnitud (mapa de calor de las 48 combinaciones) -> rampa de un solo tono.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from utils.formato import num, pct
from utils.modelo import (
    panel_minimo,
    NOMBRE_BALANCEO,
    NOMBRE_MODELO,
    NOMBRE_OPTIMIZADOR,
    ORDEN_MODELOS,
    UMBRAL,
    cargar_resumen,
)
from utils.theme import (
    AXIS,
    BG_CARD,
    COLOR_GBM,
    COLOR_LGG,
    COLOR_MARK,
    GRID,
    INK,
    INK_MUTED,
    INK_SOFT,
    SEQ_BLUE,
    TONOS_MODELO,
    TRAZO_MODELO,
    apply_theme,
)

SIMBOLO_MODELO = {"logistica": "circle", "random_forest": "square", "xgboost": "diamond"}

METRICAS = {
    "auc": ("AUC", "Área bajo la curva ROC", True),
    "recall": ("Recall GBM", "Proporción de glioblastomas detectados (umbral 0,5)", True),
    "f1": ("F1", "Media armónica de precisión y recall (umbral 0,5)", True),
    "brier": ("Brier", "Error cuadrático de las probabilidades (más bajo es mejor)", False),
}

BALANCEOS = ["ninguno", "smote", "adasyn", "class_weight"]
OPTIMIZADORES = ["grid", "random", "bayesiano", "genetico"]


def _nombre_variable(variable: str) -> str:
    return {"Age_at_diagnosis": "Edad", "Race_binary": "Grupo racial (no blanco)",
            "Gender": "Género (mujer)"}.get(variable, variable)


# --------------------------------------------------------------------------- #
# Pestaña Modelos · diseño y resultados de la validación anidada
# --------------------------------------------------------------------------- #
def fig_combinaciones(metrica: str = "auc") -> go.Figure:
    """Mapa de calor de las 48 combinaciones (filas modelo · balanceo).

    Una rampa de un solo tono: es una magnitud. En Brier, donde menos es mejor,
    la rampa se invierte para que "más oscuro" siga queriendo decir "mejor".
    """
    combos = cargar_resumen()["combinaciones"]
    etiqueta, _, mas_es_mejor = METRICAS[metrica]
    filas = [(m, b) for m in ORDEN_MODELOS for b in BALANCEOS]
    z, texto, hover = [], [], []
    for m, b in filas:
        fila_z, fila_t, fila_h = [], [], []
        for o in OPTIMIZADORES:
            c = next(c for c in combos if (c["modelo"], c["balanceo"], c["metodo"]) == (m, b, o))
            media, de = c[f"{metrica}_media"], c[f"{metrica}_de"]
            fila_z.append(media)
            fila_t.append(num(media, 3))
            fila_h.append(
                f"<b>{NOMBRE_MODELO[m]}</b> · {NOMBRE_BALANCEO[b]} · {NOMBRE_OPTIMIZADOR[o]}"
                f"<br>{etiqueta}: {num(media, 3)} ± {num(de, 3)} (10 pliegues)"
                f"<br>AUC {num(c['auc_media'], 3)} · recall {num(c['recall_media'], 3)}"
                f"<br>Tiempo de la validación anidada: {num(c['t_total_s'], 0)} s"
            )
        z.append(fila_z)
        texto.append(fila_t)
        hover.append(fila_h)

    escala = SEQ_BLUE if mas_es_mejor else SEQ_BLUE[::-1]
    valores = np.array(z)
    fig = go.Figure(go.Heatmap(
        z=valores,
        x=[NOMBRE_OPTIMIZADOR[o] for o in OPTIMIZADORES],
        y=[f"{NOMBRE_MODELO[m]} · {NOMBRE_BALANCEO[b]}" for m, b in filas],
        text=texto, texttemplate="%{text}", textfont=dict(size=12),
        customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        colorscale=[[i / (len(escala) - 1), c] for i, c in enumerate(escala)],
        xgap=2, ygap=2,
        colorbar=dict(title=dict(text=etiqueta, side="right"), thickness=12,
                      outlinewidth=0, tickfont=dict(color=INK_MUTED, size=11)),
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
    fig.update_xaxes(side="top", showline=False, ticks="")
    return apply_theme(fig, height=520, margin=dict(l=10, r=10, t=40, b=10))


def fig_pliegues() -> go.Figure:
    """AUC de los 10 pliegues externos de cada modelo final, con su media.

    Enseña lo que una cifra sola esconde: la variación de un pliegue a otro es
    mucho mayor que la diferencia entre modelos.
    """
    modelos = cargar_resumen()["modelos"]
    fig = go.Figure()
    for i, m in enumerate(ORDEN_MODELOS):
        aucs = modelos[m]["auc_pliegues"]
        fig.add_trace(go.Scatter(
            x=aucs, y=[NOMBRE_MODELO[m]] * len(aucs), mode="markers",
            marker=dict(size=11, color=TONOS_MODELO[m], symbol=SIMBOLO_MODELO[m],
                        line=dict(color=BG_CARD, width=2), opacity=0.85),
            hovertemplate="Pliegue %{customdata}: AUC %{x:.3f}<extra>" + NOMBRE_MODELO[m] + "</extra>",
            customdata=list(range(1, len(aucs) + 1)), showlegend=False,
        ))
        media = float(np.mean(aucs))
        fig.add_trace(go.Scatter(
            x=[media, media], y=[NOMBRE_MODELO[m]] * 2, mode="markers",
            marker=dict(symbol="line-ns", size=30, line=dict(width=3, color=INK)),
            hovertemplate=f"Media: {num(media, 3)}<extra>{NOMBRE_MODELO[m]}</extra>",
            showlegend=False,
        ))
    fig.update_xaxes(title_text="AUC del pliegue externo", showgrid=True, gridcolor=GRID,
                     tickformat=".2f")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return apply_theme(fig, height=260, margin=dict(l=10, r=20, t=20, b=50))


def fig_anytime(modelo: str = "xgboost") -> go.Figure:
    """Curvas anytime: cuánto le falta a cada optimizador para el mejor del bloque."""
    curvas = cargar_resumen()["anytime"][modelo]
    tonos = {"grid": "#8796AC", "random": "#52698A", "bayesiano": "#1E4E86", "genetico": "#6B7F9C"}
    trazos = {"grid": "dot", "random": "dash", "bayesiano": "solid", "genetico": "dashdot"}
    fig = go.Figure()
    t = list(range(1, 25))
    for o in OPTIMIZADORES:
        y = [max(v, 1e-4) for v in curvas[o]]
        fig.add_trace(go.Scatter(
            x=t, y=y, mode="lines", name=NOMBRE_OPTIMIZADOR[o],
            line=dict(color=tonos[o], width=2.5 if o == "bayesiano" else 2, dash=trazos[o]),
            hovertemplate="Evaluación %{x}: le faltan %{y:.4f} de AUC<extra>"
                          + NOMBRE_OPTIMIZADOR[o] + "</extra>",
        ))
    marcas = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    fig.update_yaxes(type="log", title_text="Brecha con el mejor (AUC, escala log)",
                     tickvals=marcas, ticktext=[num(v, 4).rstrip("0").rstrip(",") for v in marcas])
    fig.update_xaxes(title_text="Evaluaciones de hiperparámetros", dtick=4)
    return apply_theme(fig, height=360, margin=dict(l=10, r=10, t=40, b=50))


# --------------------------------------------------------------------------- #
# Pestaña Modelos · evaluación en el conjunto de prueba
# --------------------------------------------------------------------------- #
def fig_roc() -> go.Figure:
    """Curvas ROC de los tres modelos finales en los 168 pacientes de prueba."""
    modelos = cargar_resumen()["modelos"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                             line=dict(color=AXIS, dash="dot", width=1), hoverinfo="skip"))
    for m in ORDEN_MODELOS:
        roc = modelos[m]["roc"]
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"], mode="lines", line_shape="hv",
            name=f"{NOMBRE_MODELO[m]} · AUC {num(modelos[m]['auc_test'], 3)}",
            line=dict(color=TONOS_MODELO[m], width=2.2, dash=TRAZO_MODELO[m]),
            hovertemplate="Falsos positivos %{x:.0%} · GBM detectados %{y:.0%}<extra>"
                          + NOMBRE_MODELO[m] + "</extra>",
        ))
    fig.update_xaxes(title_text="Tasa de falsos positivos (LGG marcados como GBM)",
                     range=[0, 1], tickformat=".0%", constrain="domain")
    fig.update_yaxes(title_text="Recall de GBM", range=[0, 1.02], tickformat=".0%",
                     scaleanchor="x", scaleratio=1)
    fig.update_layout(legend=dict(orientation="v", x=0.98, xanchor="right", y=0.04,
                                  yanchor="bottom", bgcolor="rgba(255,255,255,0.85)"))
    return apply_theme(fig, height=420, margin=dict(l=10, r=10, t=20, b=50))


def matriz_confusion(modelo: str, umbral: float = UMBRAL) -> dict:
    """Matriz de confusión y métricas del modelo en prueba para un umbral dado."""
    test = cargar_resumen()["test"]
    y = np.asarray(test["y"])
    pred = (np.asarray(test[modelo]) >= umbral).astype(int)
    vp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    vn = int(((pred == 0) & (y == 0)).sum())
    return {"vp": vp, "fn": fn, "fp": fp, "vn": vn,
            "recall": vp / max(vp + fn, 1), "especificidad": vn / max(vn + fp, 1),
            "precision": vp / max(vp + fp, 1), "accuracy": (vp + vn) / len(y)}


def fig_matriz_confusion(modelo: str, umbral: float = UMBRAL) -> go.Figure:
    """Matriz de confusión en prueba. Cada fila se normaliza a su propio 100 %."""
    mc = matriz_confusion(modelo, umbral)
    cuentas = np.array([[mc["vn"], mc["fp"]], [mc["fn"], mc["vp"]]])
    pct_fila = cuentas / cuentas.sum(1, keepdims=True)
    etiquetas = [["LGG bien clasificados", "LGG marcados como GBM"],
                 ["GBM que se escapan", "GBM detectados"]]
    texto = [[f"<b>{cuentas[i, j]}</b><br>{pct(pct_fila[i, j] * 100, 0)}" for j in range(2)]
             for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=pct_fila, x=["Predicho LGG", "Predicho GBM"], y=["Real LGG", "Real GBM"],
        text=texto, texttemplate="%{text}", textfont=dict(size=15),
        customdata=etiquetas, hovertemplate="%{customdata}: %{z:.0%} de su fila<extra></extra>",
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
        zmin=0, zmax=1, showscale=False, xgap=3, ygap=3,
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="")
    fig.update_xaxes(side="top", showline=False, ticks="")
    return apply_theme(fig, height=300, margin=dict(l=10, r=10, t=40, b=10))


def fig_calibracion(modelo: str) -> go.Figure:
    """Diagrama de confiabilidad: fuera de pliegue (671) y prueba (168)."""
    datos = cargar_resumen()["modelos"][modelo]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                             line=dict(color=AXIS, dash="dot", width=1), hoverinfo="skip"))
    for fuente, clave, trazo, simbolo in [
        ("Fuera de pliegue (n = 671)", "calibracion_oof", "solid", "circle"),
        ("Prueba (n = 168)", "calibracion_test", "dash", "square"),
    ]:
        c = datos[clave]
        fig.add_trace(go.Scatter(
            x=c["predicha"], y=c["observada"], mode="lines+markers", name=fuente,
            line=dict(color=TONOS_MODELO[modelo], width=2, dash=trazo),
            marker=dict(size=[6 + n ** 0.5 for n in c["n"]], symbol=simbolo,
                        line=dict(color=BG_CARD, width=1.5)),
            customdata=c["n"],
            hovertemplate="Predicha %{x:.0%} · observada %{y:.0%}<br>%{customdata} pacientes"
                          "<extra>" + fuente + "</extra>",
        ))
    fig.update_xaxes(title_text="Probabilidad de GBM predicha", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title_text="Frecuencia observada de GBM", range=[0, 1], tickformat=".0%")
    fig.update_layout(legend=dict(orientation="v", x=0.02, xanchor="left", y=0.98,
                                  yanchor="top", bgcolor="rgba(255,255,255,0.85)"))
    return apply_theme(fig, height=360, margin=dict(l=10, r=10, t=20, b=50))


# --------------------------------------------------------------------------- #
# Pestaña Modelos · interpretación
# --------------------------------------------------------------------------- #
def fig_importancia(n: int = 10) -> go.Figure:
    """Importancia global (porcentaje del |SHAP| medio) en XGBoost y en la logística."""
    filas = cargar_resumen()["importancia"][:n][::-1]
    fig = go.Figure()
    for clave, nombre in [("logistica", NOMBRE_MODELO["logistica"]), ("xgboost", NOMBRE_MODELO["xgboost"])]:
        fig.add_trace(go.Bar(
            y=[_nombre_variable(f["variable"]) for f in filas], x=[f[clave] for f in filas],
            orientation="h", name=nombre, marker_color=TONOS_MODELO[clave],
            text=[pct(f[clave], 1) for f in filas], textposition="outside",
            textfont=dict(size=11, color=INK_SOFT), cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f} % de la importancia total<extra>" + nombre + "</extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08)
    fig.update_xaxes(title_text="% del total de |SHAP| medio", showgrid=True, gridcolor=GRID,
                     range=[0, 58])
    fig.update_yaxes(showgrid=False)
    _redondear(fig)
    return apply_theme(fig, height=440, margin=dict(l=10, r=30, t=40, b=50))


def fig_panel_reducido() -> go.Figure:
    """AUC del modelo con las k variables más importantes (ranking dentro de cada pliegue)."""
    resumen = cargar_resumen()
    panel = resumen["panel"]
    k, auc, de = panel["k"], panel["auc"], panel["de"]
    completo = auc[-1]
    # La curva es la del mejor procedimiento en validación cruzada (el
    # notebook la entrena con él): ahora Random Forest, antes XGBoost
    mejor = max(resumen["modelos"], key=lambda m: resumen["modelos"][m]["auc_cv_media"])
    k_min = panel_minimo(resumen)["k"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k + k[::-1], y=[a + d for a, d in zip(auc, de)] + [a - d for a, d in zip(auc, de)][::-1],
        fill="toself", fillcolor="rgba(30,78,134,0.10)", line=dict(width=0),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_hline(y=completo, line=dict(color=INK_MUTED, dash="dash", width=1),
                  annotation_text=f"Las 23 variables: {num(completo, 3)}",
                  annotation_position="bottom right",
                  annotation_font=dict(size=11, color=INK_MUTED))
    fig.add_trace(go.Scatter(
        x=k, y=auc, mode="lines+markers", line=dict(color=TONOS_MODELO[mejor], width=2.5),
        marker=dict(size=8, color=TONOS_MODELO[mejor], line=dict(color=BG_CARD, width=2)),
        customdata=de, showlegend=False,
        hovertemplate="%{x} variables: AUC %{y:.3f} ± %{customdata:.3f}<extra></extra>",
    ))
    # El panel mínimo, con la regla de un error estándar, como en el libro
    fig.add_vline(x=k_min, line=dict(color=INK_MUTED, dash="dot", width=1),
                  annotation_text=f"Panel mínimo: {k_min} variables",
                  annotation_position="top right",
                  annotation_font=dict(size=11, color=INK_MUTED))
    fig.add_annotation(x=4, y=auc[k.index(4)], text="IDH1 + edad + TP53 + PTEN",
                       showarrow=True, arrowhead=0, arrowcolor=INK_MUTED, ax=60, ay=40,
                       font=dict(size=12, color=INK))
    fig.update_xaxes(title_text="Número de variables (las k más importantes)", tickvals=k)
    fig.update_yaxes(title_text="AUC (media ± DE, 10 pliegues)", tickformat=".2f",
                     range=[0.8, 0.97])
    return apply_theme(fig, height=360, margin=dict(l=10, r=10, t=20, b=50))


# --------------------------------------------------------------------------- #
# Pestaña Predicción
# --------------------------------------------------------------------------- #
def fig_probabilidades(probas: dict[str, float]) -> go.Figure:
    """Probabilidad de GBM que da cada modelo, con el umbral de decisión.

    Cada barra lleva el color del grado que predice: aquí el azul y el coral sí
    hablan de un grado, que es su único significado en el tablero.
    """
    nombres = [NOMBRE_MODELO[m] for m in ORDEN_MODELOS]
    valores = [probas[m] for m in ORDEN_MODELOS]
    colores = [COLOR_GBM if v >= UMBRAL else COLOR_LGG for v in valores]
    fig = go.Figure(go.Bar(
        y=nombres, x=valores, orientation="h", marker_color=colores, width=0.55,
        text=[f"{pct(v * 100, 0)} · {'GBM' if v >= UMBRAL else 'LGG'}" for v in valores],
        textposition="outside", cliponaxis=False,
        textfont=dict(size=13, color=INK),
        hovertemplate="%{y}: P(GBM) = %{x:.1%}<extra></extra>",
    ))
    fig.add_vline(x=UMBRAL, line=dict(color=INK_MUTED, dash="dot", width=1.2),
                  annotation_text="umbral 50 %", annotation_position="top",
                  annotation_font=dict(size=11, color=INK_MUTED))
    fig.update_xaxes(range=[0, 1.18], tickvals=[0, 0.25, 0.5, 0.75, 1], tickformat=".0%",
                     showgrid=True, gridcolor=GRID, title_text="Probabilidad de GBM")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    _redondear(fig)
    return apply_theme(fig, height=230, margin=dict(l=10, r=10, t=30, b=45))


def fig_explicacion(explicacion: dict, paciente: dict, n: int = 8) -> go.Figure:
    """Cascada de contribuciones (SHAP) de una predicción.

    Parte del valor base (la predicción media de la cohorte de entrenamiento) y
    suma, de mayor a menor peso, lo que aporta cada variable hasta llegar a la
    predicción de este paciente. Coral = empuja hacia GBM; azul = hacia LGG.
    """
    contrib = explicacion["contribuciones"]
    orden = sorted(contrib, key=lambda v: abs(contrib[v]), reverse=True)
    top, resto = orden[:n], orden[n:]

    def etiqueta(v):
        if v == "Age_at_diagnosis":
            return f"Edad = {num(float(paciente['Age_at_diagnosis']), 0)} años"
        if v == "Gender":
            return f"Género = {'mujer' if int(paciente.get('Gender', 0)) == 1 else 'hombre'}"
        if v == "Race_binary":
            return f"Grupo racial = {'blanco' if int(paciente.get('Race', 0)) == 0 else 'no blanco'}"
        return f"{v} {'mutado' if int(paciente.get(v, 0)) == 1 else 'no mutado'}"

    y = [etiqueta(v) for v in top]
    x = [contrib[v] for v in top]
    if resto:
        y.append(f"Otras {len(resto)} variables")
        x.append(sum(contrib[v] for v in resto))
    base = explicacion["base"]
    final = base + sum(contrib.values())
    escala = explicacion["escala"]
    if escala == "log-odds":
        fmt = lambda v: num(v, 2, signo=True)
        corte, unidades = 0.0, "log-odds de GBM"
        rotulo_final = f"este paciente: {num(final, 2)} (P = {pct(100 / (1 + np.exp(-final)), 0)})"
        rotulo_base = f"base {num(base, 2)}"
    else:
        fmt = lambda v: num(v * 100, 1, signo=True) + " pp"
        corte, unidades = UMBRAL, "probabilidad de GBM"
        rotulo_final = f"este paciente: {pct(final * 100, 0)}"
        rotulo_base = f"base {pct(base * 100, 0)}"

    # Solo las contribuciones son barras: la cascada arranca en el valor base y
    # termina en la predicción, que se marcan con líneas. Una barra desde el cero
    # para la base no significaría nada en log-odds.
    fig = go.Figure(go.Waterfall(
        orientation="h", base=base, y=y, x=x, measure=["relative"] * len(x),
        text=[fmt(v) for v in x], textposition="outside", cliponaxis=False,
        textfont=dict(size=12, color=INK_SOFT),
        increasing=dict(marker=dict(color=COLOR_GBM)),
        decreasing=dict(marker=dict(color=COLOR_LGG)),
        connector=dict(line=dict(color=AXIS, width=1)),
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    acumulado = base + np.concatenate([[0], np.cumsum(x)])
    lo = min(acumulado.min(), corte)
    hi = max(acumulado.max(), corte)
    margen = 0.22 * (hi - lo if hi > lo else 1)
    fig.update_xaxes(range=[lo - margen, hi + margen],
                     title_text=f"Contribución acumulada ({unidades})", showgrid=True,
                     gridcolor=GRID, zeroline=False,
                     tickformat=".0%" if escala != "log-odds" else None)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    # En log-odds el cero es justo el umbral de decisión (P = 50 %): a la
    # derecha el modelo dice GBM y a la izquierda LGG.
    fig.add_vline(x=corte, line=dict(color=INK_MUTED, dash="dot", width=1.2))
    fig.add_vline(x=base, line=dict(color=AXIS, dash="dash", width=1.2))
    fig.add_vline(x=final, line=dict(color=INK, width=2))
    for x_, texto, ancla, y_ in [(base, rotulo_base, "center", 1.13),
                                 (final, rotulo_final, "center", 1.05),
                                 (corte, "umbral 50 %", "center", -0.02)]:
        fig.add_annotation(x=x_, y=y_, xref="x", yref="paper", text=texto, showarrow=False,
                           xanchor=ancla, yanchor="bottom" if y_ > 0 else "top",
                           font=dict(size=11, color=INK if texto.startswith("este") else INK_MUTED),
                           bgcolor="rgba(255,255,255,0.85)")
    return apply_theme(fig, height=150 + 34 * len(y),
                       margin=dict(l=10, r=20, t=60, b=60), showlegend=False)


def fig_contexto(modelo: str, proba: float) -> go.Figure:
    """Dónde cae la predicción entre los 168 pacientes de prueba de cada grado."""
    test = cargar_resumen()["test"]
    y = np.asarray(test["y"])
    p = np.asarray(test[modelo])
    rng = np.random.default_rng(0)
    fig = go.Figure()
    for grado, nombre, color in [(0, "LGG", COLOR_LGG), (1, "GBM", COLOR_GBM)]:
        valores = p[y == grado]
        fig.add_trace(go.Scatter(
            x=valores, y=grado + rng.uniform(-0.22, 0.22, len(valores)), mode="markers",
            name=f"{nombre} real (prueba)",
            marker=dict(size=8, color=color, opacity=0.65, line=dict(color=BG_CARD, width=1)),
            hovertemplate=f"{nombre} real · P(GBM) = %{{x:.0%}}<extra></extra>",
        ))
    fig.add_vline(x=proba, line=dict(color=INK, width=2.5),
                  annotation_text=f"este perfil: {pct(proba * 100, 0)}",
                  annotation_position="top", annotation_font=dict(size=12, color=INK))
    fig.add_vline(x=UMBRAL, line=dict(color=INK_MUTED, dash="dot", width=1))
    fig.update_xaxes(range=[-0.02, 1.02], tickformat=".0%", title_text="Probabilidad de GBM",
                     showgrid=True, gridcolor=GRID)
    fig.update_yaxes(tickvals=[0, 1], ticktext=["LGG real", "GBM real"], range=[-0.5, 1.5],
                     showgrid=False)
    fig.update_layout(legend=dict(orientation="h", y=1.18, x=0, xanchor="left", yanchor="bottom"))
    return apply_theme(fig, height=300, margin=dict(l=10, r=10, t=70, b=40))


def _redondear(fig: go.Figure, radio: int = 4) -> None:
    try:
        fig.update_traces(marker_cornerradius=radio, selector=dict(type="bar"))
    except (ValueError, TypeError):
        pass


# --------------------------------------------------------------------------- #
# Predicción · dentro del Random Forest
# --------------------------------------------------------------------------- #
def pregunta_nodo(variable: str, umbral: float, menor: bool = False) -> str:
    """La pregunta que hace un nodo, en palabras (`menor`: corte "<", XGBoost)."""
    if variable == "Age_at_diagnosis":
        return f"¿Edad {'<' if menor else '≤'} {num(umbral, 1)} años?"
    if variable == "Gender":
        return "¿Es mujer?"
    if variable == "Race_binary":
        return "¿Grupo racial no blanco?"
    return f"¿{variable} mutado?"


def respuesta_nodo(variable: str, izquierda: bool) -> str:
    """Sí / No según la rama que toma el paciente.

    En la edad la rama izquierda es "≤ corte" (sí); en las binarias la
    izquierda es el 0 (no mutado, hombre, blanco), es decir, "no".
    """
    si = izquierda if variable == "Age_at_diagnosis" else not izquierda
    return "Sí" if si else "No"


def fig_votacion(probas: np.ndarray, indice: int | None = None) -> go.Figure:
    """Probabilidad de GBM que da cada uno de los árboles del bosque.

    La predicción del bosque es la media de estas probabilidades. La forma del
    histograma dice cuánto acuerdo hay: una sola montaña pegada a un extremo es
    un bosque unánime; dos montañas, un bosque dividido.
    """
    probas = np.asarray(probas)
    media = float(probas.mean())
    bins = dict(start=0, end=1.0001, size=0.05)
    fig = go.Figure()
    for mascara, nombre, color in [(probas < UMBRAL, "Árboles que dicen LGG", COLOR_LGG),
                                   (probas >= UMBRAL, "Árboles que dicen GBM", COLOR_GBM)]:
        fig.add_trace(go.Histogram(
            x=probas[mascara], xbins=bins, name=f"{nombre} ({int(mascara.sum())})",
            marker=dict(color=color, line=dict(color=BG_CARD, width=1)),
            hovertemplate="P(GBM) entre %{x}: %{y} árboles<extra></extra>",
        ))
    fig.add_vline(x=UMBRAL, line=dict(color=INK_MUTED, dash="dot", width=1.2))
    fig.add_vline(x=media, line=dict(color=INK, width=2.5),
                  annotation_text=f"media del bosque: {pct(media * 100, 0)}",
                  annotation_position="top", annotation_font=dict(size=12, color=INK))
    if indice is not None:
        fig.add_vline(x=float(probas[indice]), line=dict(color=COLOR_MARK, dash="dash", width=1.6),
                      annotation_text=f"árbol {indice + 1}", annotation_position="bottom right",
                      annotation_font=dict(size=11, color=COLOR_MARK))
    fig.update_layout(barmode="overlay",
                      legend=dict(orientation="h", y=1.16, x=0, xanchor="left", yanchor="bottom"))
    fig.update_xaxes(range=[-0.02, 1.02], tickformat=".0%", showgrid=False,
                     title_text="Probabilidad de GBM que da el árbol")
    fig.update_yaxes(title_text="Número de árboles", showgrid=True, gridcolor=GRID)
    _redondear(fig, 3)
    return apply_theme(fig, height=300, margin=dict(l=10, r=10, t=70, b=45))


def fig_arbol(arbol: dict, camino: dict, predictoras: list[str], umbral_legible) -> go.Figure:
    """Un árbol completo del bosque con el camino del paciente resaltado.

    El árbol entero se dibuja tenue, para que se vea su tamaño; solo el camino
    del paciente lleva tinta y rótulos. Las hojas se colorean por el grado que
    predicen (azul LGG, coral GBM), que es lo único que significan esos colores.
    """
    izq, der = arbol["izq"], arbol["der"]
    var, umbral = arbol["var"], arbol["umbral"]
    valor, cob = arbol["valor"], arbol["cobertura"]

    # Posiciones: cada hoja ocupa una columna, en orden de izquierda a derecha;
    # cada nodo interno se centra sobre sus dos hijos.
    x, prof, contador = {}, {}, [0]

    def colocar(n, d):
        prof[n] = d
        if izq[n] == -1:
            x[n] = contador[0]
            contador[0] += 1
            return
        colocar(int(izq[n]), d + 1)
        colocar(int(der[n]), d + 1)
        x[n] = (x[int(izq[n])] + x[int(der[n])]) / 2

    colocar(0, 0)
    en_camino = set(camino["nodos"])

    logodds = camino.get("escala") == "log-odds"
    menor = camino.get("menor", False)

    def texto_hover(n):
        if logodds:
            if izq[n] == -1:
                return f"Hoja · suma {num(valor[n], 3, signo=True)} a los log-odds"
            variable = predictoras[var[n]]
            return pregunta_nodo(variable, umbral_legible(variable, float(umbral[n])), menor)
        base = f"{num(cob[n], 0)} casos de entrenamiento · {pct(valor[n] * 100, 0)} GBM"
        if izq[n] == -1:
            return f"Hoja · {base}"
        variable = predictoras[var[n]]
        return f"{pregunta_nodo(variable, umbral_legible(variable, float(umbral[n])))}<br>{base}"

    fig = go.Figure()
    ex, ey, cx, cy = [], [], [], []
    for n in x:
        if izq[n] == -1:
            continue
        for h in (int(izq[n]), int(der[n])):
            destino = (cx, cy) if (n in en_camino and h in en_camino) else (ex, ey)
            destino[0].extend([x[n], x[h], None])
            destino[1].extend([-prof[n], -prof[h], None])
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=AXIS, width=1),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", line=dict(color=INK, width=3),
                             hoverinfo="skip", showlegend=False))

    for grupo in ("internos", "hojas"):
        nodos = [n for n in x if (izq[n] == -1) == (grupo == "hojas")]
        colores, tamanos, opac = [], [], []
        for n in nodos:
            activo = n in en_camino
            if grupo == "hojas":
                hacia_gbm = valor[n] > 0 if logodds else valor[n] >= UMBRAL
                colores.append(COLOR_GBM if hacia_gbm else COLOR_LGG)
                tamanos.append(18 if activo else 8)
                opac.append(1 if activo else 0.45)
            else:
                colores.append(INK if activo else AXIS)
                tamanos.append(11 if activo else 6)
                opac.append(1)
        fig.add_trace(go.Scatter(
            x=[x[n] for n in nodos], y=[-prof[n] for n in nodos], mode="markers",
            marker=dict(color=colores, size=tamanos, opacity=opac,
                        line=dict(color=BG_CARD, width=1)),
            text=[texto_hover(n) for n in nodos], hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    # Rótulos solo en el camino: la pregunta y lo que responde el paciente.
    centro = (max(x.values()) + min(x.values())) / 2
    for paso in camino["pasos"]:
        n = paso["nodo"]
        a_la_derecha = x[n] <= centro
        fig.add_annotation(
            x=x[n], y=-prof[n],
            text=f"{pregunta_nodo(paso['variable'], paso['umbral'], menor)} "
                 f"<b>{respuesta_nodo(paso['variable'], paso['izquierda'])}</b>",
            showarrow=False, xanchor="left" if a_la_derecha else "right",
            xshift=10 if a_la_derecha else -10,
            font=dict(size=11, color=INK), bgcolor="rgba(255,255,255,0.88)",
        )
    hoja = camino["hoja"]
    fig.add_annotation(
        x=x[hoja], y=-prof[hoja],
        text=(f"<b>{num(camino['proba'], 3, signo=True)} log-odds</b>" if logodds
              else f"<b>{pct(camino['proba'] * 100, 0)} GBM</b>"),
        showarrow=False, yshift=-20, font=dict(size=12, color=INK),
        bgcolor="rgba(255,255,255,0.88)",
    )
    profundidad = max(prof.values())
    fig.update_xaxes(visible=False, range=[min(x.values()) - 1, max(x.values()) + 1])
    fig.update_yaxes(visible=False, range=[-profundidad - 0.8, 0.5])
    return apply_theme(fig, height=110 + 52 * (profundidad + 1),
                       margin=dict(l=10, r=10, t=10, b=10), showlegend=False)



def _sigmoide(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def fig_suma_xgb(base: float, aportes: np.ndarray, indice: int | None = None) -> go.Figure:
    """Cómo se construye la predicción de XGBoost, árbol a árbol.

    Arranca en el valor base y cada árbol suma su aporte en log-odds; el eje
    muestra la probabilidad equivalente para que se lea igual que el resto.
    """
    acumulado = base + np.concatenate([[0.0], np.cumsum(aportes)])
    proba = _sigmoide(acumulado)
    x = np.arange(len(acumulado))
    final = float(proba[-1])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=proba, mode="lines", line=dict(color=INK, width=2.5),
        hovertemplate="Tras %{x} árboles: P(GBM) = %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[x[-1]], y=[final], mode="markers",
        marker=dict(size=12, color=COLOR_GBM if final >= UMBRAL else COLOR_LGG,
                    line=dict(color=BG_CARD, width=2)),
        hoverinfo="skip",
    ))
    fig.add_hline(y=UMBRAL, line=dict(color=INK_MUTED, dash="dot", width=1.2),
                  annotation_text="umbral 50 %", annotation_position="bottom right",
                  annotation_font=dict(size=11, color=INK_MUTED))
    fig.add_annotation(x=0, y=float(proba[0]), text=f"base {pct(proba[0] * 100, 0)}",
                       showarrow=False, xanchor="left", yanchor="bottom", yshift=6,
                       xshift=12, font=dict(size=11, color=INK_MUTED),
                       bgcolor="rgba(255,255,255,0.85)")
    fig.add_annotation(x=x[-1], y=final, text=f"<b>{pct(final * 100, 0)}</b>",
                       showarrow=False, xanchor="right", yanchor="bottom", yshift=10,
                       font=dict(size=13, color=INK))
    if indice is not None:
        fig.add_vline(x=indice + 1, line=dict(color=COLOR_MARK, dash="dash", width=1.4),
                      annotation_text=f"árbol {indice + 1}", annotation_position="top",
                      annotation_font=dict(size=11, color=COLOR_MARK))
    fig.update_xaxes(title_text="Árboles sumados", range=[-4, len(acumulado) + 4],
                     showgrid=False)
    fig.update_yaxes(title_text="Probabilidad de GBM", range=[-0.02, 1.02],
                     tickformat=".0%", showgrid=True, gridcolor=GRID)
    return apply_theme(fig, height=320, margin=dict(l=10, r=20, t=30, b=45),
                       showlegend=False)


def fig_curva_logistica(z: float) -> go.Figure:
    """La curva en S de la regresión logística, con este perfil y la cohorte.

    La suma de efectos (log-odds) se convierte en probabilidad con la sigmoide.
    Los puntos de fondo son los 168 pacientes de prueba, en el color de su grado
    real; la marca grande es el perfil del formulario.
    """
    test = cargar_resumen()["test"]
    y = np.asarray(test["y"])
    p = np.clip(np.asarray(test["logistica"], dtype=float), 1e-4, 1 - 1e-4)
    zt = np.log(p / (1 - p))
    lim = float(max(8.0, np.ceil(np.abs(np.concatenate([zt, [z]])).max() + 0.5)))
    malla = np.linspace(-lim, lim, 240)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=malla, y=_sigmoide(malla), mode="lines",
                             line=dict(color=AXIS, width=3), hoverinfo="skip",
                             showlegend=False))
    for grado, nombre, color in [(0, "LGG", COLOR_LGG), (1, "GBM", COLOR_GBM)]:
        sel = zt[y == grado]
        fig.add_trace(go.Scatter(
            x=sel, y=_sigmoide(sel), mode="markers", name=f"{nombre} real (prueba)",
            marker=dict(size=7, color=color, opacity=0.55, line=dict(color=BG_CARD, width=1)),
            hovertemplate=f"{nombre} real · suma %{{x:.2f}} · P = %{{y:.0%}}<extra></extra>",
        ))
    pz = float(_sigmoide(z))
    fig.add_trace(go.Scatter(
        x=[z], y=[pz], mode="markers", name="Este perfil",
        marker=dict(size=16, color=INK, line=dict(color=BG_CARD, width=2.5)),
        hovertemplate="Este perfil · suma %{x:.2f} · P = %{y:.1%}<extra></extra>",
    ))
    fig.add_annotation(x=z, y=pz, text=f"<b>este perfil: {pct(pz * 100, 0)}</b>",
                       showarrow=False, xanchor="right" if z > 0 else "left",
                       xshift=-14 if z > 0 else 14, font=dict(size=12, color=INK),
                       bgcolor="rgba(255,255,255,0.85)")
    fig.add_vline(x=0, line=dict(color=INK_MUTED, dash="dot", width=1))
    fig.add_hline(y=UMBRAL, line=dict(color=INK_MUTED, dash="dot", width=1))
    fig.update_xaxes(title_text="Suma de efectos (log-odds de GBM)", range=[-lim, lim],
                     zeroline=False, showgrid=False)
    fig.update_yaxes(title_text="Probabilidad de GBM", range=[-0.03, 1.03],
                     tickformat=".0%", showgrid=True, gridcolor=GRID)
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=0, xanchor="left",
                                  yanchor="bottom"))
    return apply_theme(fig, height=340, margin=dict(l=10, r=20, t=50, b=45))
