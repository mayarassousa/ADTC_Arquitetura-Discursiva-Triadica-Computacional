# Base RAG V8

Formato CSV esperado:

| coluna | descrição |
|---|---|
| `id` | identificador estável |
| `tema` | palavras-chave/descrição temática |
| `fragmento` | evidência contextual |
| `fonte_sugerida` | referência ou nota de proveniência |
| `origem_contexto` | opcional: base curada, memória discursiva etc. |
| `tipo_contexto` | opcional: evento, citação, memória, política etc. |

A V8 acrescenta automaticamente o treino como memória textual sem rótulo e seis seeds de memória discursiva. O desempenho desses seeds deve ser auditado; eles não são considerados contexto pertinente por definição.
