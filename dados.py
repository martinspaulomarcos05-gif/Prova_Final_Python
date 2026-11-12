import pandas as pd


def carregar_dados():
    df = pd.read_csv("steam.csv")

    print(f"Total de linhas lidas do CSV: {len(df)}")

    if 'owners' in df.columns:
        # Tenta extrair o primeiro número da faixa de owners de forma mais segura
        df['owners_num'] = (
            df['owners']
            .astype(str)
            .str.extract(r'(\d+[\d,]*)')[0]
            .str.replace(',', '')
        )
    else:
        print("AVISO: Coluna 'owners' não encontrada!")

    df['owners_num'] = pd.to_numeric(df['owners_num'], errors='coerce')
    print(f"Linhas com owners_num válido: {df['owners_num'].dropna().shape[0]}")

    df = df.dropna(subset=['owners_num'])

    if len(df) == 0:
        raise ValueError(
            "ERRO: Todas as linhas foram zeradas no tratamento de 'owners'! Verifique o formato da coluna no CSV.")

    limiar = df['owners_num'].median()
    df['target'] = (df['owners_num'] > limiar).astype(int)

    features = ['price', 'genres', 'platforms', 'required_age']
    df_modelo = df[features + ['target']].dropna()
    print(f"Linhas após o dropna das features: {len(df_modelo)}")

    generos_endcode = df_modelo['genres'].str.get_dummies(sep=';')
    plataformas_endcode = df_modelo['platforms'].str.get_dummies(sep=';')

    X = pd.concat([df_modelo[['price', 'required_age']], generos_endcode, plataformas_endcode], axis=1)
    y = df_modelo['target']

    return X, y


if __name__ == "__main__":
    X, y = carregar_dados()
    print("Dados carregados com sucesso!")
    print(f"Formato de X (features): {X.shape}")
    print(f"Dados de y (target): {y.shape[0]} linhas")