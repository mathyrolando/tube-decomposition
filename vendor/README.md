# Código de terceros

| Qué | Dónde | Origen | Commit | Cambios locales |
| --- | --- | --- | --- | --- |
| GPTQ | `vendor/gptq/` (copia versionada) | https://github.com/IST-DASLab/gptq | `2d65066` | `datautils.py`: `wikitext` → `Salesforce/wikitext` |
| ADMM-Q | `external/admm-q/` (clon, no incluido) | https://github.com/mazumder-lab/admm-q | `251647a` | ninguno |
| AWQ | `external/llm-awq/` (clon, no incluido) | https://github.com/mit-han-lab/llm-awq | `d6e797a` | `llm-awq-local.patch` |

Para reconstruir `external/` desde cero:

```bash
git clone https://github.com/mazumder-lab/admm-q external/admm-q && git -C external/admm-q checkout 251647a
git clone https://github.com/mit-han-lab/llm-awq external/llm-awq && git -C external/llm-awq checkout d6e797a
git -C external/llm-awq apply ../../vendor/llm-awq-local.patch
```
