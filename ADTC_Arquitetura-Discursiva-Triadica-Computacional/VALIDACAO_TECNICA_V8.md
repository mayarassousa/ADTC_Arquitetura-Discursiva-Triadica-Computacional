# Validação técnica do pacote V8

- Compilação estática dos módulos Python: concluída sem erros.
- Testes automatizados: **3 aprovados**.
  - trava anti-vazamento;
  - cálculo de Precision@k e Hit@k;
  - fluxo lexical mínimo do RAG integrado.
- Teste de integração sintético: pipeline executado do carregamento à geração da planilha final.

A execução completa com BGE-M3, cross-encoder e dados integrais do IDPT depende do ambiente com os modelos e os arquivos autorizados.
