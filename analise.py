import os
import pandas as pd
import matplotlib.pyplot as plt


def carregar_dados_brutos():
    # Lê diretamente o CSV original para ter acesso às colunas textuais (genres, owners, price)
    df = pd.read_csv('steam.csv')
    return df


def graficos_generos(df):
    generos = df['genres'].dropna().str.split(';').explode()
    contagem_generos = generos.value_counts().head(10)

    plt.figure(figsize=(10, 6))
    contagem_generos.plot(kind='bar', color='skyblue')

    plt.title('10 gêneros mais frequentes na Steam')
    plt.xlabel('Gêneros')
    plt.ylabel('Quantidade de jogos')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Salva para o site e mostra na tela
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/grafico_generos.png')
    plt.show()


def graficos_precos(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df.loc[df['price'] <= 50, 'price'].dropna(), bins=30, color='salmon', edgecolor='black')

    plt.title('Distribuição dos preços dos jogos')
    plt.xlabel('Preço (US$)')
    plt.ylabel('Quantidade de jogos')
    plt.tight_layout()

    # Salva para o site e mostra na tela
    plt.savefig('static/grafico_precos.png')
    plt.show()


def graficos_preco_owners(df):
    dados = df[['owners', 'price']].dropna().copy()

    dados['limite_inferior'] = dados['owners'].apply(
        lambda faixa: int(faixa.split('-')[0].replace(',', '').strip())
    )

    precos_por_faixa = dados.groupby(['limite_inferior', 'owners'])['price'].median().reset_index()
    precos_por_faixa = precos_por_faixa.sort_values('limite_inferior')

    plt.figure(figsize=(12, 6))
    plt.bar(precos_por_faixa['owners'], precos_por_faixa['price'], color='mediumseagreen')

    plt.title('Preço mediano por faixa de proprietários')
    plt.xlabel('Faixa de proprietários')
    plt.ylabel('Preço mediano (US$)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Salva para o site e mostra na tela
    plt.savefig('static/grafico_owners.png')
    plt.show()


if __name__ == '__main__':
    df = carregar_dados_brutos()

    graficos_generos(df)
    graficos_precos(df)
    graficos_preco_owners(df)