"""Genera Reporte_HojaDeTrabajo2.pdf a partir de resultados.json y figuras/ (ejecutar después del notebook)."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

REPO_URL = "https://github.com/iancumes/HojaDeTrabajo2DeepLearning"

R = json.loads(Path("resultados.json").read_text(encoding="utf-8"))
FIG = Path("figuras")

ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=8.3, leading=10.2, alignment=TA_JUSTIFY, spaceAfter=2.5)
H1 = ParagraphStyle("h1", parent=B, fontName="Helvetica-Bold", fontSize=10.5, leading=12.5, spaceBefore=5, spaceAfter=2.5, textColor=colors.HexColor("#1f3b6f"), alignment=0)
CAP = ParagraphStyle("cap", parent=B, fontSize=7.2, leading=8.6, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
TIT = ParagraphStyle("t", parent=B, fontName="Helvetica-Bold", fontSize=13, leading=16, alignment=TA_CENTER, spaceAfter=1)
SUB = ParagraphStyle("s", parent=B, fontSize=8.5, alignment=TA_CENTER, spaceAfter=4)

S = []
p = lambda t: S.append(Paragraph(t, B))
h = lambda t: S.append(Paragraph(t, H1))

def fig(nombre, ancho_cm, texto):
    img = Image(str(FIG / nombre))
    r = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = ancho_cm * cm, ancho_cm * cm * r
    S.append(KeepTogether([img, Paragraph(texto, CAP), Spacer(1, 3)]))

esc, cost, ent, cor = R["escalamiento"], R["costo"], R["entropia"], R["coref"]
top = cor["top"][0]
gp = R["gpt2_primer_token"]

S.append(Paragraph("Hoja de trabajo 2 — Transformers y mecanismos de atención", TIT))
S.append(Paragraph("CC3092 Deep Learning y Sistemas Inteligentes · Ian Cumes · Universidad del Valle de Guatemala", SUB))

# ---------------------------------------------------------------- 2
h("1. De las RNN a la atención")
p("<b>Cuello de botella seq2seq.</b> En el encoder-decoder con RNN el encoder comprime toda la oración de entrada en un único "
  "vector de tamaño fijo (su último estado oculto) y el decoder debe generar la salida solo a partir de él. Con oraciones largas, "
  "la información de los primeros tokens se diluye o se sobrescribe y la calidad (p. ej. BLEU) cae al crecer la longitud. "
  "<b>Bahdanau et al. (2015)</b> conservan <i>todos</i> los estados del encoder y, en cada paso del decoder, calculan un vector de "
  "contexto distinto como promedio ponderado de ellos; los pesos (alineamiento suave) dependen del estado actual del decoder, "
  "de modo que el modelo «vuelve a mirar» la parte relevante de la entrada.")
p("<b>Aditiva vs multiplicativa.</b> Bahdanau puntúa con una pequeña red: e = v<sup>T</sup> tanh(W<sub>1</sub>s + W<sub>2</sub>h). "
  "Luong (2015) usa un producto punto (s<sup>T</sup>h) o bilineal (s<sup>T</sup>Wh). Ambas tienen complejidad asintótica similar, pero "
  "el producto punto se expresa como <b>una sola multiplicación de matrices</b> QK<sup>T</sup>, operación altamente optimizada en GPU/TPU "
  "(GEMM, tensor cores), sin la no linealidad elemento a elemento ni el tensor intermedio de la aditiva; por eso es más eficiente en "
  "hardware moderno (la aditiva solo rendía mejor con d<sub>k</sub> grande sin escalar, problema que corrige el factor 1/√d<sub>k</sub>).")
p("<b>Q, K, V como diccionario suave.</b> En un diccionario clásico, una consulta coincide exactamente con una llave y devuelve su valor. "
  "En la atención, la consulta se compara con <i>todas</i> las llaves (similitud q·k), la softmax convierte las similitudes en pesos que "
  "suman 1 y se devuelve la mezcla ponderada de los valores: una búsqueda diferenciable. En <b>self-attention</b> Q, K y V son "
  "proyecciones lineales (W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>) de la <i>misma</i> secuencia; en <b>cross-attention</b> Q sale del "
  "decoder y K, V de la salida del encoder.")
p("<b>Limitaciones de las RNN que elimina el Transformer.</b> (i) El cómputo secuencial h<sub>t</sub>=f(h<sub>t−1</sub>,x<sub>t</sub>) "
  "impide paralelizar a lo largo del tiempo; la atención procesa todos los tokens a la vez. (ii) Dos tokens a distancia n están separados "
  "por O(n) pasos en una RNN y por O(1) en la atención, lo que facilita las dependencias de largo alcance. (iii) El gradiente ya no "
  "atraviesa n multiplicaciones por la misma matriz recurrente, lo que reduce el desvanecimiento/explosión (junto con las conexiones residuales).")

# ---------------------------------------------------------------- 3
h("2. La arquitectura Transformer")
p(f"<b>Scaled dot-product attention.</b> Si q<sub>i</sub>, k<sub>i</sub> son independientes con media 0 y varianza 1, "
  f"q·k = Σ q<sub>i</sub>k<sub>i</sub> es suma de d<sub>k</sub> términos de media 0 y varianza 1, así que <b>Var(q·k) = d<sub>k</sub></b>. "
  f"Los logits crecen como √d<sub>k</sub>, la softmax se satura (casi one-hot) y su jacobiano diag(p)−pp<sup>T</sup> tiende a 0: los "
  f"gradientes se desvanecen. Dividir entre √d<sub>k</sub> devuelve la varianza a 1. Lo verificamos numéricamente (fig. 1): con "
  f"d<sub>k</sub>=1024 la varianza sin escalar es {esc['var_raw'][-1]:.0f} (escalada {esc['var_esc'][-1]:.2f}), la probabilidad máxima "
  f"promedio sube a {esc['pmax_raw'][-1]:.2f} (vs {esc['pmax_esc'][-1]:.2f}) y la norma del jacobiano cae a {esc['grad_raw'][-1]:.3f} "
  f"(vs {esc['grad_esc'][-1]:.3f}, constante con escalamiento).")
p("<b>Multi-head attention.</b> Con h cabezas de dimensión d<sub>model</sub>/h cada cabeza aprende su propio subespacio y su propio "
  "patrón de atención (sintaxis, posición, correferencia…); una sola cabeza promediaría todo en una única distribución. El número de "
  "parámetros <b>no cambia</b> (4·d<sub>model</sub><sup>2</sup> + sesgos): para d<sub>model</sub>=512 obtuvimos 1,050,624 parámetros con h = 1, 2, 4 u 8; "
  "el costo de cómputo también es prácticamente el mismo.")
p("<b>Codificación posicional.</b> La self-attention es equivariante a permutaciones: si se permutan los tokens, la salida se permuta "
  "igual, por lo que no distingue «perro muerde hombre» de «hombre muerde perro». El artículo suma PE(pos,2i)=sin(pos/10000<sup>2i/d</sup>) "
  "y PE(pos,2i+1)=cos(·): frecuencias geométricas, sin parámetros, y PE(pos+k) es función lineal de PE(pos), lo que facilita atender por "
  "posición relativa. Los <b>embeddings aprendidos</b> (BERT, GPT-2) son una tabla entrenable: flexibles pero no extrapolan más allá de la "
  "longitud máxima vista. <b>RoPE</b> (Su et al., 2021) rota q y k según su posición, de modo que q·k depende solo de la distancia relativa; "
  "se aplica en cada capa y es el estándar actual (LLaMA, Mistral), con mejor extrapolación vía interpolación de frecuencias.")
p("<b>Bloque completo.</b> Cada subcapa (atención y FFN) va envuelta en una conexión residual x + Sublayer(x), que crea un camino directo "
  "para el gradiente, y en LayerNorm, que estabiliza la escala de las activaciones. La <b>FFN por posición</b> (dos lineales con ReLU/GELU, "
  "d<sub>ff</sub> ≈ 4·d<sub>model</sub>) se aplica a cada token por separado y aporta la mayor parte de los parámetros y la no linealidad. "
  "<b>Post-LN</b> (original): LN(x + Sublayer(x)); el gradiente pasa por LN en cada capa, requiere <i>warm-up</i> y es inestable en "
  "modelos profundos. <b>Pre-LN</b>: x + Sublayer(LN(x)); la ruta residual queda limpia, entrena estable sin warm-up y escala a cientos de "
  "capas (Xiong et al., 2020), a costa de un rendimiento final a veces ligeramente menor.")
p(f"<b>Complejidad.</b> Tiempo O(n<sup>2</sup>·d) y memoria O(n<sup>2</sup>) por la matriz QK<sup>T</sup>. En nuestra medición (fig. 2) la pendiente "
  f"log-log del tiempo fue {cost['pendiente']:.2f} (≈2) y la matriz de atención pasa de 0.5 MB (n=128) a 512 MB (n=4096) por capa. "
  "Alternativas: <b>FlashAttention</b> (Dao et al., 2022) calcula la atención exacta por bloques en SRAM sin materializar n×n: memoria O(n), "
  "mismo resultado, pero el tiempo sigue siendo O(n<sup>2</sup>) y requiere kernels específicos de hardware. <b>Atención por ventanas</b> "
  "(Longformer, Mistral) restringe cada token a w vecinos (+ tokens globales): O(n·w), sacrificando interacción directa a larga distancia. "
  "<b>Atención lineal</b> (Katharopoulos et al., 2020) reemplaza la softmax por un kernel φ(q)φ(k)<sup>T</sup> y reordena el producto: O(n), "
  "pero aproxima la atención y suele perder precisión en tareas de recuperación exacta.")
fig("escalamiento.png", 16.5, "<b>Fig. 1.</b> Efecto de d<sub>k</sub> con q, k ~ N(0, I) y 16 llaves: varianza de q·k, probabilidad máxima de la softmax y norma del jacobiano, con y sin el factor 1/√d<sub>k</sub>.")
fig("costo.png", 12.5, "<b>Fig. 2.</b> Costo de la self-attention en CPU (8 cabezas, d=64): tiempo de la versión ingenua vs <font face='Courier'>F.scaled_dot_product_attention</font> y memoria de la matriz n×n.")

# ---------------------------------------------------------------- 4
h("3. Atención en PyTorch y Hugging Face")
p("<b>F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal, scale)</b>: atención fusionada que elige automáticamente un "
  "kernel (FlashAttention, memory-efficient o matemático). <i>attn_mask</i> es booleana (True = se puede atender) o aditiva (float, −inf "
  "bloquea); <i>dropout_p</i> aplica dropout a los pesos (usar 0 en evaluación); <i>is_causal=True</i> aplica la máscara triangular sin "
  "construirla; <i>scale</i> reemplaza el 1/√d<sub>k</sub> por defecto. No devuelve los pesos.")
p("<b>nn.MultiheadAttention(embed_dim, num_heads, batch_first, kdim, vdim)</b>: módulo completo con proyecciones Q/K/V "
  "(<i>in_proj_weight</i> 3E×E) y de salida. embed_dim debe ser divisible entre num_heads; <i>batch_first=True</i> usa (B, n, E); "
  "<i>kdim/vdim</i> permiten que K y V vengan de otra dimensión (cross-attention). En forward(query, key, value): <i>key_padding_mask</i> "
  "(B, n<sub>k</sub>) ignora el padding, <i>attn_mask</i> (n<sub>q</sub>×n<sub>k</sub> o B·h×n<sub>q</sub>×n<sub>k</sub>) impone p. ej. causalidad, "
  "<i>need_weights</i> devuelve los pesos y <i>average_attn_weights</i> decide si se promedian entre cabezas o se devuelven por cabeza.")
p("<b>nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, norm_first)</b>: bloque atención + FFN con residuales; "
  "<i>dim_feedforward</i> es d<sub>ff</sub>, <i>norm_first=True</i> lo convierte en Pre-LN (por defecto Post-LN). "
  "<b>nn.Transformer.generate_square_subsequent_mask(n)</b> devuelve una matriz float n×n con 0 en y bajo la diagonal y −inf encima: la "
  "máscara causal aditiva.")
p("<b>Hugging Face.</b> Con <i>output_attentions=True</i> el modelo devuelve <i>outputs.attentions</i>: una tupla con <b>una entrada por "
  "capa</b> (12 en BERT-base y GPT-2), cada una de forma <b>(batch, cabezas, n, n)</b>, donde la fila i son los pesos softmax de la query i "
  "sobre todas las keys. Requiere <i>attn_implementation=\"eager\"</i>, porque SDPA/Flash no materializan los pesos.")
p(f"<b>Verificaciones de la implementación.</b> Nuestra atención desde cero coincide con <font face='Courier'>F.scaled_dot_product_attention</font> "
  f"(error máx {R['err_sdpa']:.1e}, con y sin máscara causal); nuestro multi-head con los pesos copiados reproduce "
  f"<font face='Courier'>nn.MultiheadAttention</font> (error {R['err_mha']:.1e}, mismos 4,224 parámetros); y, aplicada a los embeddings de "
  f"BERT con las proyecciones de su capa 0, reproduce exactamente las atenciones que devuelve el modelo (error {R['err_bert_capa0']:.1e}).")

# ---------------------------------------------------------------- 5
h("4. Visualización de atención en BERT y GPT-2")
cb = {n: (l, hh, s) for n, l, hh, s in R["cabezas_bert"]}
cg = {n: (l, hh, s) for n, l, hh, s in R["cabezas_gpt2"]}
p(f"Para cada cabeza calculamos la masa media de atención hacia: token anterior, siguiente, [CLS]/primer token, [SEP], puntuación y el "
  f"mismo token, y elegimos la cabeza más representativa de cada patrón en capas distintas (oración: <i>«The quick brown fox jumps over "
  f"the lazy dog, and then it falls asleep.»</i>). En <b>BERT</b> (fig. 3) aparecen patrones muy nítidos: L{cb['siguiente'][0]}-H{cb['siguiente'][1]} "
  f"atiende al token siguiente ({cb['siguiente'][2]:.2f}), L{cb['anterior'][0]}-H{cb['anterior'][1]} al anterior ({cb['anterior'][2]:.2f}), "
  f"L{cb['[CLS]'][0]}-H{cb['[CLS]'][1]} casi todo a [CLS] ({cb['[CLS]'][2]:.2f}), L{cb['[SEP]'][0]}-H{cb['[SEP]'][1]} a [SEP] ({cb['[SEP]'][2]:.2f}) y "
  f"L{cb['puntuación'][0]}-H{cb['puntuación'][1]} a la coma y el punto ({cb['puntuación'][2]:.2f}). La atención a la misma palabra es débil en BERT "
  f"(máx. {cb['misma palabra'][2]:.2f}). {R['bert_cabezas_sep_cls']} de 144 cabezas ponen más del 50 % de su masa en [CLS]/[SEP], concentradas en las capas 4–9, "
  f"igual que reportan Clark et al. (2019). En <b>GPT-2</b> (fig. 4) hay una cabeza de token anterior casi perfecta "
  f"(L{cg['anterior'][0]}-H{cg['anterior'][1]}, {cg['anterior'][2]:.2f}), una de la misma palabra en la capa 0 ({cg['misma palabra'][2]:.2f}) y cabezas "
  f"que mandan prácticamente todo al primer token (L{cg['primer token'][0]}-H{cg['primer token'][1]}, {cg['primer token'][2]:.2f}).")
fig("bert_cabezas.png", 14.2, "<b>Fig. 3.</b> BERT: seis cabezas de capas distintas (eje y = query, eje x = key) con el patrón detectado y su puntaje.")
fig("gpt2_cabezas.png", 17.5, "<b>Fig. 4.</b> GPT-2: cabezas representativas. Toda la masa queda en o bajo la diagonal por la máscara causal.")
p(f"<b>Correferencia de «it».</b> Con «…because it was too <i>tired</i>» vs «…too <i>wide</i>», buscamos cabezas con "
  f"a(it→animal) > a(it→street) en el primer caso y lo contrario en el segundo. {cor['n_cabezas']} de 144 cabezas cumplen la condición. "
  f"La mejor es <b>capa {top['capa']}, cabeza {top['cabeza']}</b> (fig. 5): con <i>tired</i> «it» atiende a animal {top['tired_animal']:.2f} vs "
  f"street {top['tired_street']:.2f}, y con <i>wide</i> a animal {top['wide_animal']:.2f} vs street {top['wide_street']:.2f}. Las cabezas "
  f"que cambian de opinión se agrupan en las capas medias-altas (4–10), donde la representación ya es semántica. Sin embargo, promediando "
  f"todas las cabezas la diferencia es mínima (Δ<sub>tired</sub>={cor['delta_prom'][0]:.3f}, Δ<sub>wide</sub>={cor['delta_prom'][1]:.3f}): la "
  f"información está concentrada en pocas cabezas.")
fig("bert_coref.png", 17, "<b>Fig. 5.</b> Izquierda: Δ<sub>tired</sub> − Δ<sub>wide</sub> por cabeza (rojo = la cabeza cambia de animal a street según el contexto). Derecha: fila de «it» en la mejor cabeza para ambas oraciones.")
p(f"<b>Máscara causal y attention sinks en GPT-2.</b> Con un texto de 44 tokens, el peso máximo por encima de la diagonal en las 144 cabezas fue "
  f"{R['gpt2_max_superior']:.1f} (exactamente cero): la matriz es exactamente triangular inferior y cada fila suma 1. La proporción media de atención al primer token "
  f"crece de {gp[0]:.2f} en la capa 0 a {max(gp):.2f} en las capas medias (fig. 6), muy por encima de los {R['ref_uniforme']:.2f} esperados si la atención "
  f"fuera uniforme. Al anteponer «Yesterday» el efecto es idéntico, así que depende de la <b>posición</b> y no del contenido del token. Es el "
  f"fenómeno de <b>attention sinks</b> (Xiao et al., 2023): como la softmax obliga a repartir masa 1, las cabezas que «no necesitan» "
  f"atender a nada vuelcan el sobrante en el primer token, visible para todas las queries; su valor tiene norma pequeña, por lo que actúa "
  f"como un no-op. Por eso StreamingLLM conserva siempre los primeros tokens en la caché KV: si se descartan, la perplejidad se dispara. "
  f"BERT hace lo mismo con [SEP] en capas 4–9 y con [CLS] en las primeras capas.")
fig("gpt2_sinks.png", 14, "<b>Fig. 6.</b> Izquierda: cabeza «sumidero» de GPT-2. Derecha: masa media de atención al primer token (GPT-2) y a [CLS]/[SEP] (BERT) por capa.")
mb, mg = min(range(12), key=lambda i: ent['bert_n'][i]), min(range(12), key=lambda i: ent['gpt2_n'][i])
p(f"<b>Entropía por capa</b> (fig. 7, promedio de 6 oraciones; en GPT-2 se normaliza por log del número de keys visibles). En ambos modelos "
  f"las primeras capas son las más difusas (BERT capa 0: {ent['bert_n'][0]:.2f}; GPT-2 capa 1: {ent['gpt2_n'][1]:.2f}) y la atención se "
  f"<b>concentra</b> al profundizar: mínimo en la capa {mb} de BERT ({ent['bert_n'][mb]:.2f}) y en la capa {mg} de GPT-2 ({ent['gpt2_n'][mg]:.2f}). "
  f"En GPT-2 la caída es casi monótona, impulsada por las cabezas sumidero; BERT vuelve a difundirse en las últimas capas (≈{ent['bert_n'][-1]:.2f}), "
  f"que mezclan contexto amplio para la tarea de MLM, y GPT-2 también repunta en la capa 11 ({ent['gpt2_n'][-1]:.2f}). Así, las capas profundas "
  f"atienden de forma más concentrada que las iniciales, aunque parte de esa concentración es hacia tokens sin contenido.")
fig("entropia.png", 12.5, "<b>Fig. 7.</b> Entropía media de la atención por capa (nats) y normalizada por su máximo posible.")

# ---------------------------------------------------------------- 7
h("5. Discusión")
p("<b>¿Por qué la atención permitió modelos mucho más grandes?</b> (1) Paralelismo: todos los tokens se procesan con multiplicaciones de "
  "matrices, aprovechando GPUs/TPUs durante el entrenamiento (las RNN requieren n pasos secuenciales). (2) Longitud de camino O(1) entre "
  "cualquier par de tokens frente a O(n), lo que hace aprendibles las dependencias largas y el flujo de gradiente estable con residuales y LN. "
  "(3) El costo cuadrático es el precio, pero para los contextos de entrenamiento habituales (512–4k tokens) es aceptable y altamente "
  "paralelizable, por lo que el cómputo total escala con datos y parámetros (leyes de escalamiento) mejor que con recurrencia.")
p("<b>Especialización de cabezas.</b> Observamos cabezas posicionales casi deterministas (anterior/siguiente), cabezas de puntuación, "
  f"cabezas de delimitadores y {cor['n_cabezas']} cabezas sensibles a la correferencia. Pero no todas parecen útiles: {R['bert_cabezas_sep_cls']} de 144 "
  "cabezas de BERT y la mayoría de las capas medias de GPT-2 vuelcan su masa en [SEP]/[CLS]/primer token. Esto concuerda con Michel et al. "
  "(2019) y Voita et al. (2019), que podan gran parte de las cabezas con poca pérdida de desempeño.")
p("<b>¿Los pesos de atención explican las decisiones?</b> Solo parcialmente. Jain y Wallace (2019) muestran que los pesos se correlacionan "
  "débilmente con medidas de importancia por gradiente y que existen distribuciones de atención muy distintas que producen la misma "
  "predicción. Wiegreffe y Pinter (2019) responden que la atención puede ser <i>una</i> explicación plausible si se evalúa bajo condiciones "
  "adecuadas, pero no <i>la</i> explicación única. Nuestros resultados ilustran ambas posturas: la cabeza "
  f"L{top['capa']}-H{top['cabeza']} sí «sigue» al referente correcto de «it», pero es una de 144, el promedio casi no distingue, y la mayor "
  "masa de atención va a tokens sumidero cuyos valores apenas contribuyen a la salida (atención alta ≠ influencia alta). Además ignoran "
  "las transformaciones de V, la FFN y los residuales. Para afirmar causalidad harían falta intervenciones (ablación de cabezas, gradientes, "
  "probing).")
p("<b>¿Cuándo preferir RNN o CNN?</b> Streaming y dispositivos embebidos con memoria constante por paso (reconocimiento de voz en línea, "
  "control, series de tiempo largas): una RNN/LSTM o un modelo de espacio de estados (Mamba, S4) procesa cada token en O(1). Datos pequeños "
  "con fuerte estructura local (imágenes pequeñas, señales 1D, audio): las CNN aportan sesgos inductivos de localidad e invariancia a la "
  "traslación y requieren muchos menos datos que un ViT. También cuando la latencia o el costo de inferencia en hardware limitado domina.")
p(f"<b>Documentos de 100,000 tokens.</b> La matriz de atención tendría 10<sup>10</sup> entradas por cabeza: ≈{R['mem_100k_gb']:.0f} GB en fp32 "
  "para una sola capa de 8 cabezas, y el tiempo crece cuadráticamente; además la caché KV de inferencia crece linealmente y los embeddings "
  "posicionales absolutos no extrapolan. Alternativas: FlashAttention (atención exacta con memoria lineal) + RoPE con interpolación de "
  "posiciones (YaRN); atención por ventana deslizante con tokens globales (Longformer/BigBird) o sumideros (StreamingLLM); atención lineal "
  "o modelos de espacio de estados (Mamba) en O(n); o recuperar fragmentos relevantes (RAG / chunking jerárquico) y atender solo sobre ellos.")

h("6. Conclusiones")
p("La atención resuelve el cuello de botella de seq2seq con una búsqueda diferenciable sobre todos los estados y, al eliminar la "
  "recurrencia, hace el entrenamiento paralelo y acorta a O(1) el camino entre tokens. Verificamos que nuestra implementación es "
  "numéricamente idéntica a PyTorch y a BERT, que el factor 1/√d<sub>k</sub> es necesario para evitar la saturación de la softmax y que el "
  "costo crece cuadráticamente con n. En los modelos preentrenados las cabezas se especializan (posición, puntuación, delimitadores, "
  "correferencia), muchas actúan como sumideros y la atención se concentra en capas profundas; los pesos de atención son una herramienta "
  "útil de inspección, pero no una explicación causal suficiente del comportamiento del modelo.")

h("Referencias")
refs = [
    "Vaswani, A. et al. (2017). Attention Is All You Need. arXiv:1706.03762.",
    "Bahdanau, D., Cho, K., Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. ICLR.",
    "Luong, M.-T., Pham, H., Manning, C. (2015). Effective Approaches to Attention-based Neural Machine Translation. EMNLP.",
    "Clark, K. et al. (2019). What Does BERT Look At? An Analysis of BERT's Attention. BlackboxNLP.",
    "Xiao, G. et al. (2023). Efficient Streaming Language Models with Attention Sinks. arXiv:2309.17453.",
    "Jain, S., Wallace, B. (2019). Attention is not Explanation. NAACL. · Wiegreffe, S., Pinter, Y. (2019). Attention is not not Explanation. EMNLP.",
    "Michel, P. et al. (2019). Are Sixteen Heads Really Better than One? NeurIPS. · Voita, E. et al. (2019). Analyzing Multi-Head Self-Attention. ACL.",
    "Xiong, R. et al. (2020). On Layer Normalization in the Transformer Architecture. ICML. · Su, J. et al. (2021). RoFormer (RoPE). arXiv:2104.09864.",
    "Dao, T. et al. (2022). FlashAttention. NeurIPS. · Beltagy, I. et al. (2020). Longformer. · Katharopoulos, A. et al. (2020). Transformers are RNNs. ICML.",
    "Documentación de PyTorch (torch.nn.functional.scaled_dot_product_attention, nn.MultiheadAttention, nn.Transformer) y de Hugging Face Transformers.",
]
for r in refs:
    S.append(Paragraph("• " + r, ParagraphStyle("r", parent=B, fontSize=7.4, leading=8.8, spaceAfter=0.5)))
S.append(Spacer(1, 4))
S.append(Paragraph(f"<b>Repositorio:</b> <link href='{REPO_URL}' color='blue'>{REPO_URL}</link>", B))

doc = SimpleDocTemplate("Reporte_HojaDeTrabajo2.pdf", pagesize=letter, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                        topMargin=1.3 * cm, bottomMargin=1.3 * cm, title="Hoja de trabajo 2 - Transformers y atención",
                        author="Ian Cumes")
doc.build(S, onLaterPages=lambda c, d: (c.setFont("Helvetica", 7), c.drawRightString(letter[0] - 1.6 * cm, 0.8 * cm, f"{d.page}")),
          onFirstPage=lambda c, d: (c.setFont("Helvetica", 7), c.drawRightString(letter[0] - 1.6 * cm, 0.8 * cm, "1")))
print("PDF generado")
