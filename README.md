# Descomposición por tubo

Código y resultados de la tesis de Licenciatura en Ciencia de Datos (FCEyN, UBA) *Compresión conjunta de modelos de lenguaje grandes mediante cuantización, rango bajo y esparsidad*, de Mathías Rolando, dirigida por Martín Maas.

*Tube decomposition: joint compression of LLM weights as quantization plus low rank plus sparse, where the low-rank and sparse terms are chosen to make the residual easy to quantize.*

## Qué hay acá

Cada matriz de pesos se aproxima como `W ≈ Q + L + S`, con `Q` cuantizada por grupos, `L` de rango bajo y `S` esparsa. En la descomposición por tubo, `L` y `S` se eligen para que el residuo `W - L - S` sea fácil de cuantizar. Para eso minimizan un modelo del error de cuantización, formado por el ancho del intervalo de cada grupo y por lo que se sale de él, ponderados por los pivotes de GPTQ. El cuantizador (GPTQ o ADMM-Q) corre al final. Todas las comparaciones cobran a cada término su costo real en bits.

Los archivos principales son estos.

| Archivo | Contenido |
| --- | --- |
| `tube.py` | La descomposición por tubo |
| `engine.py` | Motor por matriz: Hessiana, Cholesky, GPTQ con búsqueda de escala, ADMM-Q, rango bajo ponderado, componente esparsa y contabilidad de bits |
| `pipeline.py` | Compresión bloque por bloque del modelo y evaluación de perplejidad |
| `srr.py` | Reimplementación de SRR (arXiv 2602.02001) en el mismo motor |
| `test_*.py` | Experimentos; cada uno guarda sus resultados en `results/` |
| `run_*.sh` | Las corridas que producen las cifras de la tesis |
| `sensitivities/` | Sensibilidades K-FAC por matriz y el script que las calcula |
| `vendor/` | GPTQ vendorizado y cómo obtener ADMM-Q y AWQ (ver `vendor/README.md`) |

## Instalación

Los experimentos se corrieron en una GPU AMD Radeon RX 9060 XT de 16 GB con PyTorch 2.9.1 para ROCm 6.3. PyTorch para ROCm usa la misma interfaz que para CUDA, así que el código debería correr también en GPUs NVIDIA instalando el `torch` correspondiente.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

ADMM-Q y AWQ no se incluyen. Para usarlos hay que clonarlos en `external/` como indica `vendor/README.md`. Con OPT-1.3B en GPUs AMD hace falta `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.

## Reproducir los resultados

Los scripts se corren desde la raíz del repositorio y usan el intérprete de `$PYTHON` (por defecto, `python`). Todos retoman solos si se interrumpen. Los resultados de la tesis salen de estas corridas.

| Resultado | Script |
| --- | --- |
| Tubo contra cuantización sola con ADMM-Q, a igual costo | `run_admmH_seeds.sh`, `run_equal_bits.sh` |
| Codificación por entropía | `run_admm_entropy.sh`, `run_tube_g1024.sh`, `run_entropy_seeds.sh` |
| Comparación con SRR | `run_srr.sh`, `srr_diagnostics.py` |
| Perplejidad en C4 y divergencia KL | `run_c4kl.sh` |
| Elección del rango por validación y ruido de corrida | `run_day1.sh`, `run_val_recheck.sh` |
| OPT-1.3B | `run_opt13b.sh`, `run_opt13b_admm.sh`, `run_opt13b_admm_s0.sh` |

Cada entrada de `results/shift_comparison.json` y de los demás archivos guarda la configuración, la semilla, la perplejidad de validación y de prueba y los bits.

## Licencia

MIT, salvo `vendor/gptq`, que conserva su propia licencia.
