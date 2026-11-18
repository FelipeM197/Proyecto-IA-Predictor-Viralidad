"""Despliegue del diagnostico textual y semaforos de advertencia (Fase 8)."""
import html

import streamlit as st

ESTILO_GRAVEDAD = {
    "critica": ("🔴", "tarjeta-critica", "Critica"),
    "moderada": ("🟡", "tarjeta-alerta", "Moderada"),
}


def _valor(v: float, unidad: str) -> str:
    texto = f"{v:.2f}" if abs(v) >= 1 else f"{v:.4f}"
    return f"{texto} {unidad}".strip()


def _semaforo(deficiencias: list[dict]) -> None:
    if not deficiencias:
        st.markdown(
            "<div class='tarjeta tarjeta-verde'><div class='tarjeta-titulo'>🟢 Todo en rango</div>"
            "<div class='tarjeta-texto'>Las variables evaluadas coinciden con el perfil "
            "del contenido viral.</div></div>",
            unsafe_allow_html=True,
        )
        return

    for d in deficiencias:
        icono, clase, nivel = ESTILO_GRAVEDAD[d["gravedad"]]
        estado = "Exceso" if d["direccion"] == "exceso" else "Deficit"
        shap = "" if d["impacto_shap"] is None else f" · SHAP {d['impacto_shap']:+.4f}"
        st.markdown(
            f"<div class='tarjeta {clase}'>"
            f"<div class='tarjeta-titulo'>{icono} {html.escape(d['etiqueta'])} · {nivel}</div>"
            f"<div class='tarjeta-texto'>{estado}: <b>{_valor(d['valor'], d['unidad'])}</b> "
            f"(rango viral {_valor(d['p25'], d['unidad'])} – {_valor(d['p75'], d['unidad'])})"
            f"{shap}</div></div>",
            unsafe_allow_html=True,
        )


def mostrar_aviso_demo() -> None:
    """Aviso cuando aun no existen los artefactos entrenados de Persona 2."""
    st.markdown(
        "<div class='aviso-demo'>Modo demostracion: aun no existen el escalador o el modelo "
        "XGBoost en <code>modelos_guardados/</code>. Se muestran el perfil acustico y el "
        "diagnostico por reglas, sin prediccion de engagement.</div>",
        unsafe_allow_html=True,
    )


def mostrar_panel_recomendaciones(deficiencias: list[dict], diagnostico: dict) -> None:
    """Semaforos por deficiencia + tarjeta con los consejos de edicion sonora."""
    columna_alertas, columna_consejos = st.columns([1, 1.4], gap="large")

    with columna_alertas:
        st.markdown("#### 🚦 Diagnostico acustico")
        _semaforo(deficiencias)

    with columna_consejos:
        st.markdown("#### 🎚️ Recomendaciones de edicion")
        if diagnostico["fuente"] == "ollama":
            insignia = (
                f"<span class='insignia insignia-ok'>Generado por {diagnostico['modelo']} "
                f"en {diagnostico['duracion_seg']} s</span>"
            )
        else:
            insignia = "<span class='insignia insignia-error'>Motor de respaldo por reglas</span>"

        items = "".join(
            f"<li>{html.escape(texto)}</li>" for texto in diagnostico["recomendaciones"]
        )
        st.markdown(
            f"<div class='tarjeta tarjeta-violeta'>{insignia}"
            f"<ol class='lista-consejos'>{items}</ol></div>",
            unsafe_allow_html=True,
        )
        if diagnostico.get("motivo"):
            st.caption(f"Motivo del respaldo: {diagnostico['motivo']}")
