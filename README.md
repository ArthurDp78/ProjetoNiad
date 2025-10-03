# 🚋 Análise e Processamento de Dados do Titanic

Este projeto realiza uma análise exploratória e pipeline completo de pré-processamento para o dataset do Titanic, incluindo limpeza, integração, redução e transformação de dados.

## 📊 Sobre o Dataset

O dataset utilizado é o **Titanic-Dataset.csv**, que contém informações sobre os passageiros do navio Titanic e suas chances de sobrevivência.

### Características Principais do Dataset:

- **Fonte**: Dados históricos dos passageiros do RMS Titanic
- **Tamanho**: Aproximadamente 891 passageiros
- **Objetivo**: Predizer a sobrevivência baseada em características dos passageiros

### Variáveis Originalmente Presentes:

| Variável | Descrição | Tipo |
|----------|-----------|------|
| PassengerId | Identificador único do passageiro | Numérico |
| Survived | Sobreviveu (0 = Não, 1 = Sim) | Categórico |
| Pclass | Classe do ticket (1ª, 2ª, 3ª classe) | Categórico |
| Name | Nome completo do passageiro | Texto |
| Sex | Gênero do passageiro | Categórico |
| Age | Idade do passageiro | Numérico |
| SibSp | Número de irmãos/cônjuges a bordo | Numérico |
| Parch | Número de pais/filhos a bordo | Numérico |
| Ticket | Número do ticket | Texto |
| Fare | Tarifa paga pelo ticket | Numérico |
| Cabin | Número da cabine | Texto |
| Embarked | Porto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton) | Categórico |

## 🛠️ Funcionalidades do Projeto

### 1. **Análise Exploratória (EDA)**
- Informações gerais do dataset
- Estatísticas descritivas
- Detecção de valores ausentes
- Visualizações de distribuição

### 2. **Engenharia de Features**
- **Extração de títulos** a partir dos nomes
- **Identificação de deck** a partir do número da cabine
- **Cálculo do tamanho familiar**
- **Detecção de passageiros sozinhos**

### 3. **Limpeza de Dados**
- Imputação de valores ausentes em:
  - `Embarked` (moda)
  - `Fare` (mediana)
  - `Age` (mediana por classe e título)
- Tratamento de outliers em `Fare` usando método IQR

### 4. **Redução de Dimensionalidade**
- Remoção de colunas irrelevantes:
  - `PassengerId`, `Ticket`, `Name`, `Cabin`
- Agregação de títulos raros
- PCA opcional (95% de variância explicada)

### 5. **Pipeline de Transformação**
- Processamento separado para features numéricas e categóricas
- One-Hot Encoding para variáveis categóricas
- Padronização para variáveis numéricas

## 📁 Estrutura do Código

```python
# Principais funções:
load_data(path)              # Carrega o dataset
basic_eda(df)                # Análise exploratória
extract_title(name)          # Extrai título do nome
preprocess_data(df)          # Pipeline completo de pré-processamento
build_pipeline(use_pca)      # Constrói pipeline scikit-learn
```

## 🎯 Transformações Aplicadas

### Novas Features Criadas:

1. **Title**: Título extraído do nome (Mr, Mrs, Miss, Master, Rare)
2. **Deck**: Deck da cabine (A, B, C, D, E, F, G, T, Unknown)
3. **FamilySize**: Tamanho total da família a bordo
4. **IsAlone**: Indicador se passageiro estava sozinho

### Tratamento de Valores Ausentes:

- **Age**: Imputação baseada na mediana por classe social e título
- **Embarked**: Preenchido com o porto mais frequente
- **Fare**: Preenchido com a mediana geral
- **Cabin**: Transformado em feature 'Deck'

## 📈 Saída do Pipeline

O pipeline final produz:
- **Dados limpos** e padronizados
- **Features engineering** aplicado
- **PCA opcional** para redução dimensional
- **Matriz pronta** para modelos de machine learning

## 🚀 Como Usar

```python
# Carregar e processar dados
df = load_data("/caminho/para/Titanic-Dataset.csv")
clean_df = preprocess_data(df)

# Construir e aplicar pipeline
pipeline = build_pipeline(use_pca=True)
X_transformed = pipeline.fit_transform(clean_df.drop('Survived', axis=1))
```

## 📊 Resultados Esperados

- Dataset limpo com 10-15 features processadas
- Redução dimensional significativa com PCA
- Dados preparados para algoritmos de classificação
- Melhoria na qualidade das features para modelos preditivos

Este projeto demonstra um fluxo completo de pré-processamento de dados, desde a análise exploratória até a preparação final para modelagem preditiva.
