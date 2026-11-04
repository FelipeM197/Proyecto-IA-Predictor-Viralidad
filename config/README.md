# Fundamentacion Tecnica de Muestreo Acustico y Parametros DSP

## 1. Seleccion de 22 050 Hz y el Teorema de Nyquist-Shannon

El Teorema de Nyquist-Shannon determina que para muestrear una señal continua sin introducir distorsion por solapamiento espectral (*aliasing*), la frecuencia de muestreo $f_s$ debe superar el doble de la frecuencia maxima util $f_{\max}$:

$$f_s \ge 2 \cdot f_{\max} \implies f_{\text{Nyquist}} = \frac{f_s}{2}$$

Al fijar `SAMPLE_RATE = 22050` Hz, la frecuencia limite de Nyquist queda en $11\,025\text{ Hz}$.

### Justificacion fisica y computacional:
1. **Rango util de percepcion en redes sociales:** La voz humana concentra sus formantes principales entre $300\text{ Hz}$ y $3\,400\text{ Hz}$. Los golpes ritmicos de bateria y percusiones operan entre $60\text{ Hz}$ y $5\,000\text{ Hz}$. La informacion que determina la retencion auditiva reside antes de los $10\text{ kHz}$.
2. **Compresion de plataformas:** Redes como TikTok y YouTube Shorts aplican codecs con perdidas (AAC u Opus a 64-128 kbps) que atenuan drásticamente las frecuencias sobre los $11\text{ kHz}$. Muestrear a $44\,100\text{ Hz}$ duplicaria el consumo de memoria RAM y el costo de calculo de la FFT sin anadir senal predictiva util.
3. **Filtro antialiasing previo:** `librosa` aplica internamente un filtro pasa-bajos polifasico antes de diezmar la senal, evitando cualquier repliegue de altas frecuencias.

---

## 2. Resolucion Tiempo-Frecuencia en la STFT

La Transformada de Fourier de Tiempo Corto utiliza una ventana de 2 048 puntos y un avance temporal de 512 muestras. Bajo una tasa de $22\,050\text{ Hz}$, estos valores definen:

- **Duracion de ventana:** $\Delta t = \frac{2048}{22050} \approx 92.88\text{ ms}$.
- **Paso temporal entre tramas:** $\Delta t_{\text{salto}} = \frac{512}{22050} \approx 23.22\text{ ms}$.
- **Resolucion frecuencial:** $\Delta f = \frac{22050}{2048} \approx 10.77\text{ Hz}$ por contenedor (*bin*).

Esta configuracion equilibra el principio de incertidumbre de Gabor-Heisenberg: ofrece suficiente resolucion en frecuencia para separar componentes armonicos y mantiene precision temporal para ubicar el inicio de los ataques percusivos.
