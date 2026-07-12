# Como integrar o Pipeline V8 ao repositório atual

Este pacote foi organizado para **não substituir a v20f/Tucano** já presente no repositório. Ele acrescenta a V8 como subpacote `adtc.v8`.

## Integração manual

1. Abra a pasta raiz do repositório `ADTC_Arquitetura-Discursiva-Triadica-Computacional`.
2. Copie o conteúdo da pasta `adtc/` deste pacote para a pasta `adtc/` existente no repositório.
3. Mantenha os arquivos atuais da v20f. Em conflitos, preserve os arquivos existentes e acrescente apenas:
   - `adtc/adtc/v8/`
   - `adtc/data/v8/`
   - `adtc/notebooks/v8/`
   - `adtc/tests/v8/`
   - scripts com sufixo `_v8` e `app_v8.py`
   - `README_V8.md`, `MIGRACAO_NOTEBOOK_V8.md`, `requirements-v8.txt`, `requirements-xai.txt`, `CITATION_V8.cff` e `LICENSE_V8`.
4. Não publique os arquivos do IDPT sem observar a autorização e os termos aplicáveis.
5. Execute `pytest tests/v8 -q` antes do commit.

## Commit sugerido

```bash
git add adtc/adtc/v8 adtc/data/v8 adtc/notebooks/v8 adtc/scripts/*v8* adtc/tests/v8         adtc/README_V8.md adtc/MIGRACAO_NOTEBOOK_V8.md adtc/requirements-v8.txt         adtc/requirements-xai.txt adtc/CITATION_V8.cff adtc/LICENSE_V8
git commit -m "Adiciona Pipeline V8 modular: BERTimbau, ADTC, RAG integrado e avaliação manual"
```

## Importação

```python
from adtc.v8 import ConfigV8, PathsV8, PipelineV8
```
