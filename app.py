import flask
from flask import Flask, render_template, request
import pandas as pd
import joblib
from analise import carregar_dados_brutos, graficos_generos, graficos_preco_owners, graficos_precos
import os

app = flask.Flask(__name__)

# --- GERA OS GRÁFICOS AUTOMATICAMENTE AO INICIAR O APP ---
os.makedirs('static', exist_ok=True)
df_analise = carregar_dados_brutos()

import matplotlib.pyplot as plt
plt.ioff()

modelo = joblib.load("modelo_steam.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/graficos')
def pagina_graficos():
    return render_template('graficos.html')


@app.route('/prever', methods=['POST'])
def prever():
    try:
        # Verifica se marcou como Free-to-Play
        is_free = request.form.get('is_free')

        if is_free:
            preco = 0.0
        else:
            preco = float(request.form['price'])

        idade = int(request.form['required_age'])

        # Captura o gênero (prioriza o texto digitado, se houver; senão usa o selecionado)
        genero_digitado = request.form.get('genre_text', '').strip()
        genero_selecionado = request.form.get('genre_select', '')

        if genero_digitado:
            genero = genero_digitado.title()
        elif genero_selecionado:
            genero = genero_selecionado
        else:
            genero = "Indie"  # Valor padrão de segurança caso esqueça de preencher ambos

        novo_jogo = pd.DataFrame({
            'price': [preco],
            'genres': [genero],
            'platforms': [request.form['platforms']],
            'required_age': [idade]
        })

        from dados import carregar_dados
        X_treino, _ = carregar_dados()

        generos_novo = novo_jogo['genres'].str.get_dummies(sep=';')
        plataformas_novo = novo_jogo['platforms'].str.get_dummies(sep=';')

        df_temp = novo_jogo.drop(columns=['genres', 'platforms'])
        novo_jogo_processado = pd.concat([df_temp, generos_novo, plataformas_novo], axis=1)
        novo_jogo_processado = novo_jogo_processado.reindex(columns=X_treino.columns, fill_value=0)

        probabilidade = modelo.predict_proba(novo_jogo_processado)
        prob_alta = probabilidade[0][1]

        if preco > 40.0:
            fator_penalizacao = 1.0 / (1.0 + (preco - 40.0) / 20.0)
            prob_alta = prob_alta * fator_penalizacao

        if preco > 100.0:
            prob_alta = prob_alta * 0.10

        resultado_classe = 1 if prob_alta >= 0.5 else 0

        if resultado_classe == 1:
            status = "Alta popularidade (Sucesso de público provável)"
            cor = "green"
        else:
            status = "Baixa popularidade (Nicho ou menor volume de vendas)"
            cor = "orange"

        classificacao_idade = "Livre para todos os públicos" if idade == 0 else f"Recomendado para maiores de {idade} anos"
        texto_preco = "Gratuito (Free-to-Play)" if preco == 0 else f"${preco:.2f} USD"

        return render_template(
            'index.html',
            exibir_resultado=True,
            resultado_status=status,
            cor_status=cor,
            val_preco=texto_preco,
            val_idade=classificacao_idade,
            val_genero=genero,
            val_plataforma=request.form['platforms'].capitalize(),
        )

    except Exception as e:
        return render_template('index.html', previsao_erro=f"Erro ao processar a previsão: {str(e)}")
if __name__ == "__main__":
    app.run(debug=True)



