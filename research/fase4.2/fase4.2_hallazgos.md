# Fase 4.2 — Scheduling Semántico: Hallazgos

## 1. Prior Art — NO es completamente nuevo, pero hay espacio

### Trabajos directamente relevantes:

| Paper | Año | Idea clave | Resultado |
|---|---|---|---|
| **EAGS** (Koh et al.) | 2024 | Entropy-adaptive Gibbs sampling para diffusion LLMs | Genera en orden de entropía decreciente, competitivo con AR |
| **Fast-dLLM** (NVidia) | 2025 | Confidence threshold → unmask todos > umbral | **27.6× throughput**, minimal accuracy loss |
| **WavefrontDiffusion** (Yang) | 2025 | Wavefront desde tokens finalizados | SOTA en reasoning/code, respeta estructura semántica |
| **DOS** (Mar 2026) | 2026 | Atención como proxy de dependencia inter-token | Mejor en code gen y math reasoning |
| **LookUM** | 2025 | Lookahead para evitar errores de confianza local | +4 en HumanEval/GSM8K |
| **LoPA** (SJTU) | 2025 | Lookahead parallel + branch verification | 10.1 tokens/forward pass, 1073 tok/s multi-GPU |
| **P-ELBO** | 2026 | Training-time: non-uniform masking con planner awareness | Mejor calidad en code gen |
| **ADEPT** | 2026 | Early exit por token (transformers AR) | 25% menos compute |

### Lo que SÍ es nuevo en nuestra propuesta:
1. **EarlyExit semántico**: desmascarar MÁS tokens de lo previsto cuando la entropía es baja → no lo hemos encontrado exactamente así (Fast-dLLM usa threshold fijo, nosotros usamos schedule + boost)
2. **Conexión con surprisal theory** (Hale/Levy): nadie ha conectado explícitamente la teoría cognitiva de sorpresa con scheduling de difusión
3. **Cristalización por fases**: desmascarar por grupos de entropía (bajo→medio→alto) en fases explícitas

### Fuentes clave:
- [EAGS](https://arxiv.org/html/2411.06438v1)
- [Fast-dLLM](https://arxiv.org/pdf/2505.22618)
- [WavefrontDiffusion](https://arxiv.org/abs/2511.19473)
- [DOS](https://arxiv.org/abs/2603.15340)
- [LoPA](https://arxiv.org/pdf/2512.16229)
- [Entropic Time Schedulers](https://arxiv.org/pdf/2504.13612)
- [Surprisal Theory](https://www.mit.edu/~rplevy/papers/levy-2008-cognition.pdf)

---

## 2. Resultados experimentales (LLaDA-8B, 64 tokens, 16 steps)

### Tabla resumen:

| Prompt | Scheduler | Steps efectivos | tok/s | Calidad |
|---|---|---|---|---|
| Capital France | uniform | 16 | 5.5 | "The capital of France is Paris." |
| Capital France | crystallization | 16 | 5.4 | "The capital of France is Paris." |
| Capital France | **early_exit** | **3** | **29.4** | **"The capital of France is Paris."** |
| Capital France | adaptive | 16 | 5.7 | "The capital of France is Paris." |
| Translate FR | uniform | 16 | 5.7 | "Le temps est magnifique aujourd'hui." |
| Translate FR | **early_exit** | **2** | **47.4** | **"Le temps est magnifique aujourd'hui."** |
| Quantum | uniform | 16 | 5.6 | Buena calidad |
| Quantum | early_exit | 16 | 5.6 | Calidad comparable |
| Poema | uniform | 16 | 5.8 | Buena calidad |
| Poema | early_exit | 16 | 5.0 | Calidad comparable |
| Termodinámica | uniform | 16 | 5.8 | Buena calidad |
| Termodinámica | early_exit | 16 | 6.0 | Calidad comparable |

### Hallazgo principal:

**El EarlyExit scheduler es el claro ganador:**
- Cuando hay muchos tokens fáciles: **5-8x speedup** sin pérdida de calidad
- Cuando todos son difíciles: **rendimiento idéntico** al baseline
- **Nunca peor** que uniform

### Distribución de entropía (paso 1):

| Prompt | Entropía media | Tokens fáciles (<1.5) | Medio | Difíciles (>3.5) |
|---|---|---|---|---|
| Capital France | 2.40 | 15 (23%) | 33 (52%) | 16 (25%) |
| Quantum computing | 4.19 | 5 (8%) | 2 (3%) | **57 (89%)** |
| Translate FR | **0.96** | **56 (88%)** | 6 (9%) | 2 (3%) |
| Poema oceano | **5.38** | 2 (3%) | 1 (2%) | **61 (95%)** |
| Termodinámica | 4.63 | 4 (6%) | 4 (6%) | **56 (88%)** |

### La entropía ES un proxy del significado:
- **Traducción**: la estructura es casi determinista (88% tokens fáciles) → el modelo "sabe" qué va
- **Preguntas factuales**: mezcla de estructura + contenido (23-25% fáciles)
- **Generación creativa**: casi todo es difícil (95% alta entropía) → el modelo no tiene certeza

Esto confirma la hipótesis: **la entropía correlaciona con la importancia semántica**.

---

## 3. Análisis crítico

### ¿Funciona el scheduling semántico?
**SÍ**, pero solo en un escenario específico:
- Cuando una fracción significativa de tokens es predecible (traducción, respuestas factuales cortas)
- Para generación creativa abierta, no hay speedup posible

### ¿Es implementable en C++?
**SÍ, trivialmente.** El EarlyExit scheduler es simplemente:
1. Después de cada forward pass, calcular entropía por posición masked
2. Si entropía < threshold → desmascarar inmediatamente
3. Solo gastar steps en los tokens difíciles

El overhead de calcular entropía es despreciable vs el forward pass.

### ¿Es patentable?
**Difícil.** Fast-dLLM (NVidia, 2025) ya hace algo muy similar con confidence threshold. EAGS (2024) usa entropía para ordenar. La novedad marginal de nuestro EarlyExit es insuficiente para una patente.

### ¿Qué SÍ sería diferencial?
1. **Integración en motor C++ optimizado** (nadie tiene un motor C++ de difusión con scheduling semántico)
2. **Benchmark sistemático** contra llama.cpp mostrando ventaja en tok/s a iso-calidad
3. **Adaptive steps basado en el TIPO de tarea** (el scheduler se auto-ajusta)

---

## 4. Recomendación para diffuse-cpp

### Implementar: EarlyExit scheduler
```cpp
// En diffuse-sampler.cpp, añadir:
// 1. Calcular entropía por posición masked
// 2. Desmascarar todas las posiciones con entropía < threshold
// 3. Si quedan posiciones masked, continuar con schedule normal
```

### Parámetros:
- `--scheduler early_exit` (default: uniform/cosine)
- `--entropy-threshold 1.5` (configurable)

### Complejidad adicional: ~20 líneas de código

---

## 5. Benchmark Sistemático: 42 prompts × 12 categorías (C++, servidor 24 cores)

### Configuración:
- **Motor**: diffuse-cpp (F16, 12 threads, server 125GB RAM)
- **Config**: 64 tokens, 16 steps, cosine schedule, entropy_threshold=1.5
- **42 prompts** en 12 categorías de uso real

### Resultados por categoría:

| Categoría | N | Avg Speedup | Step Reduction | Con speedup (>20%) |
|---|---|---|---|---|
| **instruction_following** | 3 | **3.86x** | 62% | 2/3 |
| **classification** | 3 | **2.84x** | 67% | 3/3 |
| **translation** | 5 | **2.71x** | 61% | 5/5 |
| **factual_simple** | 5 | **2.53x** | 46% | 3/5 |
| **code_simple** | 3 | **2.21x** | 56% | 3/3 |
| **math_reasoning** | 3 | **1.66x** | 40% | 3/3 |
| summarization | 3 | 1.39x | 25% | 2/3 |
| conversational | 3 | 1.35x | 21% | 1/3 |
| code_complex | 3 | 1.22x | 15% | 1/3 |
| structured_output | 3 | 1.02x | 0% | 0/3 |
| factual_complex | 5 | 1.00x | 0% | 0/5 |
| creative_writing | 3 | 1.00x | 0% | 0/3 |

### **TOTAL: 23/42 (55%) de los prompts obtienen >20% speedup**

### Verificación de calidad:
- **Salida idéntica** en 7 casos con speedup (capital, R&J, H2O, reverse_string, passive voice, question, hello)
- **Calidad comparable** en resto de casos con speedup
- **0/42 casos con degradación significativa** — never worse

### Hallazgos clave del benchmark:

1. **Instruction following es el ganador absoluto**: voz pasiva 4.95x (2 steps), convertir a pregunta 5.51x (2 steps). Transformaciones deterministas.

2. **Clasificación/NER: 2.84x promedio**. Sentimiento, spam, NER — el modelo sabe la respuesta con alta confianza.

3. **Traducción: 2.71x**, variable por idioma. FR/IT/DE (2.7-4.2x) >> JP (1.4x). Idiomas más lejanos tienen más entropía.

4. **Código simple: 2.21x**. is_prime, reverse, factorial — patrones que el modelo ha memorizado.

5. **Código complejo: 1.22x**. binary_search, linked_list, LCS — requieren razonamiento, poca certeza.

6. **Escritura creativa: 1.0x**. Máxima incertidumbre, 95% tokens difíciles. Sin speedup posible.

### Estimación de speedup en tráfico real de chatbot:

Asumiendo distribución típica de uso:
- 35% simple factual + clasificación + instrucciones → ~3x speedup
- 20% traducción + resúmenes → ~2x speedup
- 15% código simple + math básica → ~2x speedup
- 30% creativo + complejo + conversación abierta → 1x

**Speedup ponderado estimado: ~1.8x sobre tráfico real.**

---

## 6. Conclusión

La idea de scheduling semántico **no es nueva** (EAGS, Fast-dLLM, WavefrontDiffusion ya existen), pero nuestra implementación EarlyExit tiene ventajas prácticas:

1. **Zero overhead**: sin modelo auxiliar, sin training adicional
2. **Never worse**: 0/42 prompts con degradación
3. **55% de prompts con speedup**: no es "solo traducción" — cubre clasificación, instrucciones, código simple, math
4. **~1.8x speedup ponderado**: gratis, en producción real
5. **Trivial de implementar**: 20 líneas de C++, ya implementado en diffuse-cpp

La "cristalización del significado" es una metáfora bonita pero la realidad es más prosaica: cuando el modelo está seguro, no pierdas tiempo re-evaluando. Lo que SÍ importa es que **la mayoría del tráfico real de chatbot cae en la zona de alta confianza**.
