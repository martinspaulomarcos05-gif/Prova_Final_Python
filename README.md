# Tema 09: Popularidade de Jogos na Steam

Projeto desenvolvido para a disciplina de Tópicos Especiais em Ciência da Computação, com o objetivo de analisar dados de jogos da Steam e utilizar técnicas de Machine Learning para classificar sua popularidade como alta ou baixa.

## Integrantes

- Marcos Paulo Martins de Andrade
- João Pedro Rosul Wojcik
- Abraão Lima Gomes

## Objetivo

O projeto busca identificar quais características podem estar relacionadas à popularidade de um jogo na Steam.

Para isso, utilizamos informações como preço, gênero, plataformas disponíveis e idade mínima recomendada. Com esses dados, desenvolvemos modelos de classificação capazes de estimar se um jogo pertence à categoria de popularidade alta ou baixa.

Também desenvolvemos uma aplicação web utilizando Flask, permitindo visualizar gráficos da análise exploratória e realizar previsões por meio de um formulário.

## Fonte dos dados

Utilizamos o dataset **Steam Store Games**, disponibilizado por Nik Davis.

O arquivo `steam.csv` contém informações sobre diversos jogos da plataforma, incluindo preço, gêneros, plataformas, classificação etária e quantidade estimada de proprietários.

Fonte: https://www.kaggle.com/datasets/nikdavis/steam-store-games

## Preparação dos dados

O arquivo `dados.py` é responsável por carregar e preparar os dados utilizados no treinamento dos modelos.

A base possui inicialmente **27.075 registros**.

Durante essa etapa, realizamos os seguintes procedimentos:

- Contagem de registros duplicados.
- Identificação de preços e idades negativas.
- Extração do limite inferior das faixas de proprietários.
- Conversão dos valores extraídos para o formato numérico.
- Remoção de registros sem valores válidos de proprietários.
- Remoção de registros com informações ausentes nas variáveis utilizadas pelo modelo.
- Transformação dos gêneros e plataformas em variáveis numéricas por meio de One-Hot Encoding.

Na execução analisada, não foram encontrados registros duplicados, preços negativos ou idades negativas.

A base utilizada pelo modelo permaneceu com 27.075 registros após o tratamento.

## Como definimos a popularidade?

O dataset apresenta a quantidade estimada de proprietários dos jogos em faixas, e não em números exatos.

Para definir a popularidade, extraímos o limite inferior dessas faixas e calculamos sua mediana utilizando a base de dados antes da divisão entre treinamento e teste.

Assim, dividimos os jogos em duas categorias:

- **Popularidade baixa (0):** jogos com quantidade estimada de proprietários menor ou igual à mediana.
- **Popularidade alta (1):** jogos com quantidade estimada de proprietários acima da mediana.

Essa classificação é utilizada como variável-alvo (`y`), ou seja, a resposta que os modelos precisam aprender a prever.

**Observação metodológica:** como o limiar é calculado antes da divisão dos dados, ele considera também as informações de proprietários dos registros que posteriormente compõem o conjunto de teste. Portanto, essa decisão deve ser considerada uma limitação metodológica na avaliação dos resultados.

## Variáveis utilizadas

As variáveis de entrada (`X`) são:

- **Preço (`price`):** preço do jogo em dólares.
- **Gêneros (`genres`):** categorias às quais o jogo pertence.
- **Plataformas (`platforms`):** sistemas operacionais disponíveis.
- **Idade mínima (`required_age`):** classificação etária recomendada.

A variável `owners` é utilizada para construir o alvo de popularidade, mas não é fornecida diretamente ao modelo.

## Machine Learning

Para realizar as previsões, utilizamos dois modelos de classificação do Scikit-learn:

### Regressão Logística

Modelo de classificação que estima a probabilidade de uma observação pertencer a determinada categoria. Foi configurado com `max_iter=1000`.

### Árvore de Decisão

Modelo que realiza classificações por meio de decisões baseadas nas características dos dados.

### Treinamento e teste

Os dados foram divididos utilizando `train_test_split`, com a seguinte configuração:

- **80% para treinamento.**
- **20% para teste.**
- **random_state=42**, para permitir a reprodução da divisão.
- **stratify=y**, para preservar a proporção das classes nos conjuntos de treinamento e teste.

Os modelos foram treinados com os dados de treinamento e avaliados com os dados de teste, que não participaram do ajuste dos modelos.

### Resultados dos modelos

Na execução utilizada para documentar os resultados, o conjunto de teste apresentou 5.415 registros, distribuídos da seguinte maneira:

- Classe 0 (popularidade baixa): 68,68%.
- Classe 1 (popularidade alta): 31,32%.

O baseline, que representa a acurácia obtida ao escolher sempre a classe mais frequente, foi de **68,68%**.

| Modelo | Acurácia |
|---|---:|
| Regressão Logística | **72,91%** |
| Árvore de Decisão | 72,00% |
| Baseline | 68,68% |

A Regressão Logística apresentou a maior acurácia e foi escolhida para utilização na aplicação.

### Matrizes de confusão

**Regressão Logística:**

```text
[[3476, 243],
 [1224, 472]]
```

**Árvore de Decisão:**

```text
[[3174, 545],
 [971, 725]]
```

Considerando a ordem das classes `[0, 1]`, as matrizes mostram os acertos e erros dos modelos.

Na Regressão Logística, por exemplo, 3.476 jogos de baixa popularidade foram classificados corretamente, enquanto 472 jogos de alta popularidade foram identificados corretamente.

Também é possível observar que o modelo apresenta dificuldade para identificar jogos de alta popularidade, classificando muitos deles incorretamente como baixa popularidade.

A escolha da Regressão Logística foi baseada na maior acurácia obtida na comparação.

*Os resultados numéricos apresentados correspondem à execução utilizada na elaboração deste documento. Eles podem variar caso o dataset ou o código sejam modificados.*

## Aplicação Web

Desenvolvemos uma aplicação utilizando Flask, HTML e Python.

A aplicação permite que o usuário informe as características de um jogo para receber uma estimativa de sua popularidade.

O formulário apresenta os seguintes campos:

- Preço do jogo.
- Opção de jogo gratuito (Free-to-Play).
- Idade mínima recomendada.
- Gênero, com possibilidade de selecionar uma opção ou digitar manualmente.
- Plataforma disponível: Windows, Mac ou Linux.

Após o preenchimento, a aplicação processa as informações, aplica a mesma organização de colunas utilizada no treinamento e executa a previsão com o modelo salvo em `modelo_steam.pkl`.

O resultado é apresentado na própria página, indicando alta ou baixa popularidade.

A aplicação também possui verificações para impedir preços e idades negativas e apresentar mensagens quando ocorre algum problema durante o processamento.

### Regra complementar de preço

Além da previsão realizada pelo modelo, implementamos uma regra complementar para jogos com preços elevados.

A partir de US$ 40, a probabilidade de popularidade alta recebe uma penalização progressiva. Para preços acima de US$ 100, é aplicada uma redução adicional.

Essa decisão foi baseada na hipótese de que preços elevados podem dificultar o acesso dos jogadores e, consequentemente, influenciar a quantidade de proprietários.

Essa regra foi definida pela equipe e não foi aprendida pelo modelo durante o treinamento.

## Análise exploratória e gráficos

O arquivo `analise.py` é responsável pela análise exploratória e pela geração de três gráficos utilizando Pandas e Matplotlib.

### 1. Gêneros mais frequentes

Apresenta os dez gêneros mais encontrados na base de dados, permitindo observar quais categorias possuem maior presença entre os jogos analisados.

### 2. Distribuição de preços

Apresenta a distribuição dos preços dos jogos, considerando os valores de até US$ 50 no histograma.

Essa visualização permite observar a concentração de jogos nas diferentes faixas de preço.

### 3. Preço mediano por faixa de proprietários

Compara o preço mediano dos jogos entre as diferentes faixas estimadas de proprietários.

Essa visualização permite observar como os preços variam entre jogos com diferentes níveis de alcance na plataforma.

Os gráficos são salvos na pasta `static/` e apresentados na página de análise exploratória da aplicação.

## Estrutura do projeto

```text
Prova_Final_Python/
│
├── app.py
├── dados.py
├── modelo.py
├── analise.py
├── modelo_steam.pkl
├── steam.csv
├── requirements.txt
│
├── static/
│   ├── grafico_generos.png
│   ├── grafico_precos.png
│   └── grafico_owners.png
│
└── templates/
    ├── index.html
    └── graficos.html
```

### Responsabilidade dos arquivos

- `app.py`: controla as rotas Flask, o formulário de previsão e a integração com o modelo.
- `dados.py`: carrega e prepara os dados para o Machine Learning.
- `modelo.py`: treina, compara e avalia os modelos, salvando o melhor resultado.
- `analise.py`: realiza a análise exploratória e gera os gráficos.
- `modelo_steam.pkl`: armazena o modelo escolhido após o treinamento.
- `steam.csv`: dataset utilizado no projeto.
- `templates/`: contém as páginas HTML.
- `static/`: armazena as imagens dos gráficos.

## Como executar o projeto

É necessário ter o Python instalado.

### 1. Instalar as dependências

No terminal, dentro da pasta do projeto, execute:

```bash
pip install -r requirements.txt
```

### 2. Treinar e avaliar os modelos

Execute:

```bash
python modelo.py
```

Esse comando carrega os dados, realiza o treinamento, compara os modelos e salva o escolhido em `modelo_steam.pkl`.

### 3. Gerar os gráficos

Execute:

```bash
python analise.py
```

Os gráficos serão gerados e armazenados na pasta `static/`.

### 4. Iniciar a aplicação

Execute:

```bash
python app.py
```

Depois, acesse pelo navegador:

```text
http://127.0.0.1:5000
```

## Limitações

O projeto possui algumas limitações:

- A quantidade de proprietários apresentada pelo dataset é estimada e organizada em faixas, não representando números exatos de vendas.
- A divisão entre popularidade alta e baixa depende da mediana calculada a partir da base utilizada.
- O cálculo do limiar utiliza a base inteira antes da divisão treino/teste, incluindo registros que posteriormente compõem o conjunto de teste.
- Os resultados dependem das informações disponíveis no dataset.
- A acurácia não representa, sozinha, a capacidade de identificar corretamente as duas classes.
- A Regressão Logística apresentou dificuldade maior em identificar jogos de alta popularidade, conforme observado na matriz de confusão.
- A penalização de preço é uma regra complementar definida pela equipe, não sendo um comportamento aprendido pelo modelo.
- A aplicação utiliza o dataset fixo do projeto e o modelo previamente treinado.

## Participação dos integrantes

**Marcos Paulo Martins de Andrade**
- Preparação dos dados em conjunto com João.
- Desenvolvimento do treinamento e comparação dos modelos de Machine Learning.
- Integração do modelo com a aplicação Flask, em conjunto com Abraão.

**João Pedro Rosul Wojcik**
- Limpeza e organização dos dados em conjunto com Marcos.
- Análise exploratória e criação dos gráficos.
- Desenvolvimento da página de visualização dos gráficos.

**Abraão Lima Gomes**
- Desenvolvimento das rotas e funcionamento da aplicação Flask.
- Criação da interface e do formulário de previsão.
- Integração do formulário com o modelo.
- Validação dos campos e apresentação dos resultados.

Todos os integrantes participaram da documentação, dos testes gerais e da preparação da apresentação do projeto.

## Considerações finais

Com este projeto, buscamos aplicar na prática os conhecimentos de Python, análise de dados, Machine Learning e desenvolvimento web.

A comparação entre dois modelos permitiu observar diferentes formas de realizar previsões e avaliar seus resultados. A aplicação Flask tornou possível utilizar o modelo de maneira simples e interativa, integrando a análise exploratória à previsão de popularidade dos jogos.
