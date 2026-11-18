"""Ensamblaje de prompts con anclaje numerico (Fase 7, Tarea 7.2).

El LLM recibe SOLO datos ya calculados (vector acustico, percentiles virales
y SHAP). No analiza audio ni inventa cifras: se limita a redactar.

Defensa en vivo (Tarea 10.2): cambiar NUM_RECOMENDACIONES de 3 a 2 modifica
la instruccion del sistema, el recorte de la respuesta y el motor de respaldo.
"""

NUM_RECOMENDACIONES = 3


def _fmt(valor: float, unidad: str = "") -> str:
    """Redondeo a 2-4 decimales segun magnitud, con unidad fisica."""
    decimales = 2 if abs(valor) >= 1 else 4
    texto = f"{valor:.{decimales}f}"
    return f"{texto} {unidad}".strip()


def armar_prompt_sistema(n_recomendaciones: int = NUM_RECOMENDACIONES) -> str:
    """Fija el rol del modelo y las restricciones de formato."""
    return (
        "Eres un productor musical tecnico especializado en retencion de audiencia "
        "en videos cortos de TikTok, Reels y YouTube Shorts.\n"
        "Reglas obligatorias:\n"
        f"1. Responde con EXACTAMENTE {n_recomendaciones} recomendaciones numeradas "
        f"(1. a {n_recomendaciones}.), una por linea.\n"
        "2. Cada recomendacion es una accion concreta de edicion de sonido "
        "(recorte, compresion, ecualizacion, tempo, volumen) con valores objetivo.\n"
        "3. Usa unicamente las cifras que te entrego. No inventes datos.\n"
        "4. Sin introducciones, saludos ni conclusiones. Maximo 40 palabras por punto.\n"
        "5. Responde en espanol."
    )


def _linea_deficiencia(d: dict) -> str:
    estado = "por encima" if d["direccion"] == "exceso" else "por debajo"
    linea = (
        f"- {d['etiqueta']}: {_fmt(d['valor'], d['unidad'])} ({estado} del rango viral "
        f"{_fmt(d['p25'], d['unidad'])} - {_fmt(d['p75'], d['unidad'])}; "
        f"mediana viral {_fmt(d['p50'], d['unidad'])}). Gravedad: {d['gravedad']}."
    )
    if d.get("impacto_shap") is not None:
        linea += f" Impacto SHAP: {d['impacto_shap']:+.4f}."
    return linea


def armar_prompt_diagnostico(
    deficiencias: list[dict],
    engagement_estimado: float | None = None,
    atributos_favorables: list[dict] | None = None,
    n_recomendaciones: int = NUM_RECOMENDACIONES,
) -> str:
    """Construye el mensaje de usuario con los datos duros de la pista."""
    partes = ["DATOS DEL AUDIO ANALIZADO"]
    if engagement_estimado is not None:
        partes.append(f"Tasa de interaccion estimada: {engagement_estimado * 100:.2f} %.")

    if deficiencias:
        partes.append("Desviaciones detectadas frente al 15 % de contenido mas viral:")
        partes.extend(_linea_deficiencia(d) for d in deficiencias)
    else:
        partes.append("Todas las variables evaluadas estan dentro del rango viral.")

    if atributos_favorables:
        nombres = ", ".join(a["variable"] for a in atributos_favorables)
        partes.append(f"Fortalezas que se deben conservar: {nombres}.")

    objetivo = (
        "corrijan las desviaciones, empezando por la de mayor gravedad"
        if deficiencias
        else "potencien aun mas el gancho y la retencion sin perder las fortalezas"
    )
    partes.append(
        f"\nTAREA: Redacta {n_recomendaciones} recomendaciones de edicion que {objetivo}."
    )
    return "\n".join(partes)
