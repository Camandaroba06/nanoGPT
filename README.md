
# Group 7: Exploring Alternative MHA–FFN Arrangements in Decoder-Only Transformers


## Resumo do Projeto

<!-- Apresentar brevemente o projeto, seu objetivo e o problema investigado. -->
Este trabalho investiga o impacto de rearranjos estruturais das camadas de Atenção Multi-Cabeça (MHA) e Redes Feed-Forward (FFN) na arquitetura Transformer decoder-only. Avaliamos empiricamente três configurações: a convencional (MHA → FFN), a invertida (FFN → MHA)  e a paralela (MHA ∥ FFN), paridade de parâmetros (~125M) e dados entre os modelos. Avaliamos o trade-off entre qualidade de modelagem (Perda e Perplexidade) e eficiência computacional (latência e speedup).

## Configuração do Ambiente

### Requisitos

<!-- Informar os requisitos necessários para executar o projeto. -->

- Python: [versão]
- [Demais dependências necessárias]

### Instalação

Clonar o repositório:

```bash
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
```

Instalar as dependências:

```bash
pip install -r requirements.txt
```

<!-- Incluir outras instruções necessárias para configurar
     o ambiente e acessar os recursos utilizados. -->

## Reprodução dos Resultados

<!-- Apresentar os comandos exatos, na ordem de execução,
     necessários para reproduzir os principais resultados do projeto. -->

### Treinamento

```bash
<comando_exato_de_treinamento>
```

### Avaliação

```bash
<comando_exato_de_avaliacao>
```

<!-- Informar como acessar os resultados gerados, caso necessário. -->
