import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

# ==========================
# 🔹 Carregando os Dados
# ==========================
def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset do Titanic"""
    try:
        df = pd.read_csv(path)
        print(f"✅ Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas")
        return df
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {path}")
        # Criar dataset de exemplo para demonstração
        print("📝 Criando dataset de exemplo...")
        return create_sample_data()

def create_sample_data():
    """Cria dados de exemplo se o arquivo não for encontrado"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.choice([0, 1], n_samples),
        'Pclass': np.random.choice([1, 2, 3], n_samples),
        'Name': [f'Passenger_{i}' for i in range(1, n_samples + 1)],
        'Sex': np.random.choice(['male', 'female'], n_samples),
        'Age': np.random.normal(30, 10, n_samples).clip(0, 80),
        'SibSp': np.random.poisson(0.5, n_samples),
        'Parch': np.random.poisson(0.5, n_samples),
        'Ticket': [f'Ticket_{i}' for i in range(1, n_samples + 1)],
        'Fare': np.random.exponential(50, n_samples),
        'Cabin': np.random.choice(['A', 'B', 'C', 'D', np.nan], n_samples),
        'Embarked': np.random.choice(['S', 'C', 'Q', np.nan], n_samples)
    }
    
    # Adicionar alguns valores ausentes
    df = pd.DataFrame(data)
    missing_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[missing_indices, 'Age'] = np.nan
    
    return df

def basic_eda(df: pd.DataFrame):
    """Análise Exploratória de Dados"""
    print("=" * 50)
    print("🔍 INFORMAÇÕES DO DATASET")
    print("=" * 50)
    print(df.info())
    
    print("\n" + "=" * 50)
    print("📊 ESTATÍSTICAS DESCRITIVAS")
    print("=" * 50)
    print(df.describe(include="all").T)
    
    print("\n" + "=" * 50)
    print("❌ VALORES AUSENTES")
    print("=" * 50)
    missing_data = df.isna().sum()
    print(missing_data[missing_data > 0])
    
    # Visualização de missing (apenas se houver valores ausentes)
    if df.isna().sum().sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isna(), cbar=True, cmap="viridis", yticklabels=False)
        plt.title("Mapa de Valores Ausentes")
        plt.tight_layout()
        plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Gráfico de valores ausentes salvo como 'missing_values.png'")
    
    # Distribuições principais (apenas se as colunas existirem)
    if 'Age' in df.columns and df['Age'].notna().sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Age'].dropna(), bins=30, kde=True)
        plt.title("Distribuição de Idade")
        plt.tight_layout()
        plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Gráfico de distribuição de idade salvo como 'age_distribution.png'")
    
    if 'Fare' in df.columns and df['Fare'].notna().sum() > 0:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df['Fare'])
        plt.title("Boxplot da Tarifa (Fare)")
        plt.tight_layout()
        plt.savefig('fare_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Boxplot de tarifa salvo como 'fare_boxplot.png'")

# ==========================
# 🔹 Funções Auxiliares
# ==========================
def extract_title(name: str) -> str:
    """Extrai o título do passageiro a partir do campo 'Name'"""
    if pd.isnull(name) or name == 'Unknown': 
        return "Unknown"
    try:
        return name.split(",")[1].split(".")[0].strip()
    except (IndexError, AttributeError):
        return "Unknown"

# ==========================
# 🔹 Limpeza + Integração + Redução
# ==========================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pré-processamento completo dos dados"""
    df = df.copy()
    
    print("\n" + "=" * 50)
    print("🔄 INICIANDO PRÉ-PROCESSAMENTO")
    print("=" * 50)

    # ---- Integração de Features ----
    print("📝 Extraindo títulos dos nomes...")
    df['Title'] = df['Name'].apply(extract_title)
    
    # Consolidar títulos raros
    rare_titles = ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle':'Miss','Ms':'Miss','Mme':'Mrs'})
    
    print("📝 Extraindo deck das cabines...")
    df['Deck'] = df['Cabin'].astype(str).str[0]
    df['Deck'] = df['Deck'].replace('n', 'Unknown')
    
    print("📝 Calculando tamanho familiar...")
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # ---- Limpeza de Dados ----
    print("🧹 Limpando valores ausentes...")
    
    # Embarked
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0] if not df['Embarked'].mode().empty else 'S')
    
    # Fare
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Age - imputação por Pclass + Title
    if 'Age' in df.columns and 'Pclass' in df.columns and 'Title' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        age_fill = df.groupby(['Pclass','Title'])['Age'].median()
        
        def fill_age(row):
            if pd.isnull(row['Age']):
                try:
                    return age_fill.loc[row['Pclass'], row['Title']]
                except:
                    return df['Age'].median()
            return row['Age']
        
        df['Age'] = df.apply(fill_age, axis=1)
    
    # Outliers de Fare (IQR)
    if 'Fare' in df.columns:
        Q1, Q3 = df['Fare'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        df['Fare'] = df['Fare'].clip(lower, upper)

    # ---- Redução ----
    print("📉 Reduzindo dimensionalidade...")
    cols_to_drop = ['PassengerId','Ticket','Name','Cabin']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df.drop(existing_cols_to_drop, axis=1, inplace=True)

    print("✅ Pré-processamento concluído!")
    return df

# ==========================
# 🔹 Transformação (Pipeline Scikit-learn)
# ==========================
def build_pipeline(use_pca=False):
    """Constrói o pipeline de transformação"""
    numeric_features = ['Age','Fare','FamilySize']
    categorical_features = ['Sex','Embarked','Title','Deck','Pclass','IsAlone']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    steps = [('preprocessor', preprocessor)]
    if use_pca:
        steps.append(('pca', PCA(n_components=0.95)))

    pipeline = Pipeline(steps)
    return pipeline

# ==========================
# 🔹 Execução Principal
# ==========================
def main():
    """Função principal"""
    print("🚀 INICIANDO ANÁLISE DO TITANIC")
    print("=" * 60)
    
    # Carregar dados
    path = "Titanic-Dataset.csv"  # Arquivo local
    df = load_data(path)

    # EDA inicial
    basic_eda(df)

    # Pré-processamento
    clean_df = preprocess_data(df)
    
    print("\n" + "=" * 50)
    print("✅ DATASET APÓS PRÉ-PROCESSAMENTO")
    print("=" * 50)
    print(f"Shape: {clean_df.shape}")
    print(f"Colunas: {list(clean_df.columns)}")
    print("\nPrimeiras 5 linhas:")
    print(clean_df.head())
    
    print("\n" + "=" * 50)
    print("📊 ESTATÍSTICAS DO DATASET PROCESSADO")
    print("=" * 50)
    print(clean_df.describe())

    # Pipeline (apenas se Survived existir)
    if 'Survived' in clean_df.columns:
        print("\n" + "=" * 50)
        print("🔧 APLICANDO PIPELINE DE TRANSFORMAÇÃO")
        print("=" * 50)
        
        pipe = build_pipeline(use_pca=True)
        X = clean_df.drop('Survived', axis=1)
        y = clean_df['Survived']

        # Verificar se há colunas suficientes para o pipeline
        required_cols = ['Age','Fare','FamilySize','Sex','Embarked','Title','Deck','Pclass','IsAlone']
        available_cols = [col for col in required_cols if col in X.columns]
        
        if len(available_cols) >= 3:  # Mínimo de colunas necessárias
            X_transformed = pipe.fit_transform(X)
            print(f"✅ Transformação concluída!")
            print(f"📏 Shape original: {X.shape}")
            print(f"📏 Shape após transformação: {X_transformed.shape}")
            
            if hasattr(pipe, 'named_steps') and 'pca' in pipe.named_steps:
                pca = pipe.named_steps['pca']
                print(f"🔢 Componentes PCA: {pca.n_components_}")
                print(f"📈 Variância explicada: {sum(pca.explained_variance_ratio_):.3f}")
        else:
            print("❌ Colunas insuficientes para o pipeline")
    else:
        print("❌ Coluna 'Survived' não encontrada para treinamento")

    print("\n" + "=" * 60)
    print("🎯 ANÁLISE CONCLUÍDA!")
    print("=" * 60)

if __name__ == "__main__":
    main()
