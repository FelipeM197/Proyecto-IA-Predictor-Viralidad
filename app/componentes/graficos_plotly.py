"""Graficos interactivos de resultados con Plotly (Fase 8, Tarea 8.2)."""
import plotly.graph_objects as go

from config import settings
from src.agente.detector_deficiencias import etiqueta_de

VERDE, VIOLETA, ROJO, GRIS = "#22c55e", "#8b5cf6", "#ef4444", "#9ca3af"


def _layout_oscuro(figura: go.Figure, alto: int) -> go.Figure:
    figura.update_layout(
        height=alto,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
    )
    return figura


def grafico_velocimetro(engagement_pct: float) -> go.Figure:
    """Velocimetro con la tasa de interaccion predicha en porcentaje."""
    tope = settings.ENGAGEMENT_MAX_GAUGE_PCT
    figura = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=engagement_pct,
            number=dict(suffix=" %", valueformat=".2f", font=dict(color=VERDE)),
            title=dict(text="Engagement estimado", font=dict(size=16)),
            gauge=dict(
                axis=dict(range=[0, tope], tickcolor=GRIS),
                bar=dict(color=VIOLETA),
                bgcolor="#1b1f2b",
                steps=[
                    dict(range=[0, tope / 3], color="#2a1f2b"),
                    dict(range=[tope / 3, 2 * tope / 3], color="#2a2a35"),
                    dict(range=[2 * tope / 3, tope], color="#1f2f2a"),
                ],
            ),
        )
    )
    return _layout_oscuro(figura, 280)


def _normalizar(variable: str, valor: float) -> float:
    minimo, maximo = settings.RANGOS_RADAR[variable]
    return max(0.0, min(1.0, (valor - minimo) / (maximo - minimo)))


def grafico_radar(vector: dict[str, float], umbrales: dict[str, dict]) -> go.Figure:
    """Radar de 5 ejes: la pista del usuario frente a la mediana viral."""
    variables = list(settings.RANGOS_RADAR)
    ejes = [etiqueta_de(v)[0] for v in variables]
    figura = go.Figure()

    if all(v in umbrales for v in variables):
        valores_virales = [_normalizar(v, umbrales[v]["p50"]) for v in variables]
        figura.add_trace(go.Scatterpolar(
            r=valores_virales + valores_virales[:1], theta=ejes + ejes[:1],
            fill="toself", name="Mediana viral", line=dict(color=VERDE),
            opacity=0.5,
        ))

    valores_usuario = [_normalizar(v, vector.get(v, 0.0)) for v in variables]
    textos = [f"{vector.get(v, 0.0):.3f} {etiqueta_de(v)[1]}" for v in variables]
    figura.add_trace(go.Scatterpolar(
        r=valores_usuario + valores_usuario[:1], theta=ejes + ejes[:1],
        text=textos + textos[:1], hovertemplate="%{theta}: %{text}<extra></extra>",
        fill="toself", name="Tu audio", line=dict(color=VIOLETA),
    ))
    figura.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#2a2f3d"),
            angularaxis=dict(gridcolor="#2a2f3d"),
        ),
        legend=dict(orientation="h", y=-0.1),
        title=dict(text="Perfil acustico", font=dict(size=16)),
    )
    return _layout_oscuro(figura, 360)


def grafico_barras_shap(contribuciones: dict[str, float]) -> go.Figure:
    """Barras horizontales con las variables de mayor impacto SHAP (|phi|)."""
    top = sorted(contribuciones.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top = sorted(top[: settings.NUM_BARRAS_SHAP], key=lambda kv: kv[1])
    figura = go.Figure(go.Bar(
        x=[valor for _, valor in top],
        y=[etiqueta_de(nombre)[0] for nombre, _ in top],
        orientation="h",
        marker_color=[VERDE if valor >= 0 else ROJO for _, valor in top],
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
    ))
    figura.update_layout(
        title=dict(text="Contribucion SHAP por variable", font=dict(size=16)),
        xaxis=dict(title="Impacto en la prediccion", gridcolor="#2a2f3d", zerolinecolor=GRIS),
        yaxis=dict(gridcolor="#2a2f3d"),
    )
    return _layout_oscuro(figura, 380)
