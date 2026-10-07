# Hoja de trabajo 2 — Transformers y mecanismos de atención

CC3092 Deep Learning y Sistemas Inteligentes · Ian Cumes

## Contenido
- `HojaDeTrabajo2_Atencion.ipynb` — notebook ejecutado y comentado:
  - atención escalada y multi-head desde cero, verificadas contra `F.scaled_dot_product_attention`, `nn.MultiheadAttention` y la capa 0 de BERT;
  - experimento del factor 1/√d_k y costo computacional O(n²);
  - mapas de calor de cabezas de BERT y GPT-2, correferencia de *it*, máscara causal, *attention sinks* y entropía por capa.
- `Reporte_HojaDeTrabajo2.pdf` — reporte (5 páginas): investigación, resultados, discusión y referencias.
- `generar_reporte.py` — genera el PDF a partir de `resultados.json` y `figuras/`.
- `figuras/` — gráficas exportadas por el notebook.

## Reproducir
```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace HojaDeTrabajo2_Atencion.ipynb
.venv/Scripts/python generar_reporte.py
```
