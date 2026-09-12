"""
Pacote src — código auxiliar do Tech Challenge Fase 2 (classificação de
qualidade de vinhos tintos, dataset WineQT).

Grupo: Efraim Oliveira, Érica Tarsis, Ricardo Moraes, Rodrigo Bernardino, Thiago Galvão
POSTECH DTAT — 2026

Módulos
    config     constantes, caminhos, estilo e auxiliares de gráfico
    data_prep  carga, deduplicação, alvo binário e divisão treino/teste
    modeling   fábrica dos 7 modelos, avaliação, validação cruzada e GridSearch
    analysis   EDA, engenharia de variáveis, avaliação visual, testes e interpretabilidade
    run_all    orquestrador que reproduz o estudo e grava tudo em results/
"""
__all__ = ["config", "data_prep", "modeling", "analysis"]
__version__ = "1.0.0"
