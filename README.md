# Prova_Final_Python

#  Tema 09: Popularidade de Jogos na Steam (Machine Learning & Flask)

Trabalho prático desenvolvido para a faculdade, aplicando engenharia de dados, Machine Learning com Regressão Logística e desenvolvimento Web full-stack com Flask.

---

##  Aluno
* **Nome:** Marcos Paulo Martins de Andrade
* **Nome:** João Pedro Rosul Wojcik
* **Nome:** Abraão Lima 
---

##  Objetivo do Projeto
Estimar e prever a popularidade de um jogo na Steam (faixa de proprietários alta ou baixa) com base em variáveis como preço, gênero, plataformas suportadas e idade mínima recomendada, acompanhado de uma análise exploratória visual integrada.

---

##  Funcionalidades do Código

O projeto está modularizado e estruturado para cumprir os requisitos acadêmicos através dos seguintes módulos:

* **`dados.py`**: Realiza a carga do dataset original da Steam (`steam.csv`), faz tratamentos e limpezas com Regex na coluna de proprietários (`owners`), e aplica o *One-Hot Encoding* nas variáveis categóricas de gêneros e plataformas.
* **`modelo.py`**: Treina o classificador de Machine Learning utilizando Regressão Logística com base nos dados processados e serializa o modelo treinado no arquivo binário `modelo_steam.pkl`.
* **`analise.py`**: Executa a análise exploratória de dados utilizando Pandas e Matplotlib, gerando automaticamente os gráficos estatísticos do projeto (gêneros mais frequentes, distribuição de preços e preço mediano por faixa de proprietários) salvos na pasta `static/`.
* **`app.py`**: Servidor web construído em Flask que gerencia as rotas da aplicação, integrando a inferência do modelo com regras de negócio (como penalidade de preço para jogos caros e suporte dinâmico a jogos *Free-to-Play*).
* **`templates/index.html`**: Página principal da interface web contendo o formulário interativo de previsão (com opções de seleção e digitação livre de gêneros, além de seleção de preço/free-to-play) e o relatório de estimativa.
* **`templates/graficos.html`**: Página web dedicada exclusivamente à exibição dos gráficos gerados pela análise exploratória dos dados.