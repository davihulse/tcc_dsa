# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 22:22:31 2024

@author: davi.hulse
"""
#%% [1] Import

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
#from pymer4.models import Lmer # estimação de modelos HLM3
from scipy import stats
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec

#%% [2] Abrir CSV

df = pd.read_csv('df_investidor10 - final.csv', na_values='-')

df = df.replace("-%", np.nan)

#%% [3] Dados atuais
#DF Atual

df = df[df['Ano'] == 'Atual']

print(df)

#%% [4] Renomear Colunas
mapa_colunas = {
    'PAPEL': 'Papel',
    'Ano': 'Ano',
    'P/L': 'PL',
    'P/RECEITA (PSR)': 'PSR',
    'P/VP': 'PVP',
    'DIVIDEND YIELD (DY)': 'DivYeld',
    'PAYOUT': 'Payout',
    'MARGEM LÍQUIDA': 'MargLiq',
    'MARGEM BRUTA': 'MargBrut',
    'MARGEM EBIT': 'MargEBIT',
    'MARGEM EBITDA': 'MargEBITDA',
    'EV/EBITDA': 'EVEBITDA',
    'EV/EBIT': 'EVEBIT',
    'P/EBITDA': 'PEBITDA',
    'P/EBIT': 'PEBIT',
    'P/ATIVO': 'PAtivo',
    'P/CAP.GIRO': 'PCapGiro',
    'P/ATIVO CIRC LIQ': 'PAtCircLiq',
    'VPA': 'VPA',
    'LPA': 'LPA',
    'GIRO ATIVOS': 'GiroAtivos',
    'ROE': 'ROE',
    'ROIC': 'ROIC',
    'ROA': 'ROA',
    'DÍVIDA LÍQUIDA / EBITDA': 'DivLiqEBITDA',
    'DÍVIDA LÍQUIDA / EBIT': 'DivLiqEBIT',
    'DÍVIDA BRUTA / PATRIMÔNIO': 'DivBrPatr',
    'PATRIMÔNIO / ATIVOS': 'PatrAtiv',
    'PASSIVOS / ATIVOS': 'PassAtiv',
    'LIQUIDEZ CORRENTE': 'LiqCorr',
    'CAGR RECEITAS 5 ANOS': 'CagrRec5a',
    'CAGR LUCROS 5 ANOS': 'CagrLuc5a',
    'Valor de mercado': 'ValorMercado',
    'Valor de firma': 'ValorFirma',
    'Patrimônio Líquido': 'PatrLiq',
    'Nº total de papeis': 'TotPapeis',
    'Ativos': 'Ativos',
    'Ativo Circulante': 'AtivCirc',
    'Dívida Bruta': 'DivBruta',
    'Dívida Líquida': 'DivLiq',
    'Disponibilidade': 'Disponibilidade',
    'Segmento de Listagem': 'SegListagem',
    'Free Float': 'FreeFloat',
    'Tag Along': 'TagAlong',
    'Liquidez Média Diária': 'LiqMedDia',
    'Setor': 'Setor',
    'Segmento': 'Segmento',
}

df = df.rename(columns=mapa_colunas)

print(df.columns)

#%% [5] Transformando strings e percentuais em números

df.columns

col_numericas = ['TotPapeis']
       
for coln in col_numericas:
    df[coln] = (df[coln]
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .astype(float))    

col_reais = [
    col for col in df.columns
    if df[col].astype(str).str.contains('R\$').any()
]
col_reais

for colr in col_reais:
    df[colr] = (df[colr]
        .str.replace('R$ ', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.strip()
        .replace('-', np.nan)            # trata '-' como NaN
        #.replace('', np.nan)             # trata string vazia como NaN
        .astype(float)
    )

col_percentuais = [
    col for col in df.columns
    if df[col].astype(str).str.contains('%').any()
]
col_percentuais

for colp in col_percentuais:
    df[colp] = (df[colp]
        .str.replace('%', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .astype(float)
        / 100)

#%% [6] Criando colunas em milhões
colunas_milhoes = ['LiqMedDia', 'TotPapeis']
for col in colunas_milhoes:
    if col in df.columns:
        df[col + '_mi'] = df[col] / 1000000

df[['TotPapeis','TotPapeis_mi']]

df.drop(colunas_milhoes, axis=1, inplace=True)


# Criando colunas em bilhões
colunas_bilhoes = ['ValorMercado', 'ValorFirma', 'PatrLiq', 'Ativos', 'AtivCirc',
                   'DivBruta', 'DivLiq', 'Disponibilidade']

for col in colunas_bilhoes:
    if col in df.columns:
        df[col + '_bi'] = df[col] / 1000000000

#Apenas para conferir as colunas de bilhões com as originais
colunas_pares = []
for col in colunas_bilhoes:
    if col in df.columns and col + '_bi' in df.columns:
        colunas_pares.extend([col, col + '_bi'])

df_bilhoes = df[colunas_pares]

#Dropar
df.drop(colunas_bilhoes, axis=1, inplace=True)


#%% [7] Incluindo Setores

setores = pd.read_excel("ClassifSetor.xlsx", sheet_name=0)
setores = setores.ffill()
setores = setores.dropna()

df['Papel_base'] = df['Papel'].str[:4]

df = df.merge(
    setores,
    left_on="Papel_base",
    right_on="CÓDIGO",
    how="left"
)

df = df.drop(columns=["CÓDIGO"])

setores.columns

mapa_setores = {
    'SETOR ECONÔMICO': 'SetorCVM',
    'SUBSETOR': 'SubsetorCVM',
    'SEGMENTO': 'SegmentoCVM',
    'NOME DE PREGÃO': 'Nome'
}

df = df.rename(columns=mapa_setores)

#%% [8] Conferência

df['SEGMENTO DE NEGOCIAÇÃO'].unique()

#df = df[df['SEGMENTO DE NEGOCIAÇÃO'] == 'Novo Mercado']


#%% [9] Verificando colunas com nulos

df.isnull().sum()

#%% [10] Dropando colunas específicas com nulos

#Qualquer valor nulo
#colunas_muitos_nulos = df.columns[df.isnull().any()].tolist()

#Acima de 'n' valores nulos
colunas_muitos_nulos = df.columns[df.isnull().sum() > 9].tolist()

print("Colunas com muitos nulos = ", colunas_muitos_nulos)
df.drop(colunas_muitos_nulos, axis=1, inplace=True)

#%% [11] Criação de variáveis de setor com médias

df['MargLiqSetor'] = df.groupby('Setor')['MargLiq'].transform('mean')
df['MargBrutSetor'] = df.groupby('Setor')['MargBrut'].transform('mean')
df['MargEBITSetor'] = df.groupby('Setor')['MargEBIT'].transform('mean')
df['MargEBITDASetor'] = df.groupby('Setor')['MargEBITDA'].transform('mean')


#%% [12] Drop colunas

#Verificar quantidade de nulos
df.isnull().sum()

df.info()

df = df.drop(columns=['Ano'])
df = df.drop(columns=['MargLiq'])
df = df.drop(columns=['MargBrut'])
df = df.drop(columns=['MargEBIT'])
df = df.drop(columns=['MargEBITDA'])
df = df.drop(columns=['SegListagem'])
df = df.drop(columns=['FreeFloat'])
df = df.drop(columns=['TagAlong'])
df = df.drop(columns=['Setor'])
df = df.drop(columns=['Segmento'])
df = df.drop(columns=['Rentab_1M'])
df = df.drop(columns=['Rentab_3M'])
df = df.drop(columns=['Rentab_1A'])
df = df.drop(columns=['Rentab_2A'])
df = df.drop(columns=['Rentab_5A'])
df = df.drop(columns=['Rentab_10A'])
df = df.drop(columns=['LiqMedDia_mi'])
df = df.drop(columns=['TotPapeis_mi'])
df = df.drop(columns=['ValorFirma_bi'])
df = df.drop(columns=['PatrLiq_bi'])
df = df.drop(columns=['Ativos_bi'])
df = df.drop(columns=['AtivCirc_bi'])
#df = df.drop(columns=['DivBruta_bi'])
df = df.drop(columns=['DivLiq_bi'])
df = df.drop(columns=['Disponibilidade_bi'])
df = df.drop(columns=['Papel_base'])
df = df.drop(columns=['SEGMENTO DE NEGOCIAÇÃO'])

df.rename(columns={'SetorCVM': 'Setor'}, inplace=True)
df.rename(columns={'SubsetorCVM': 'Subsetor'}, inplace=True)
df.rename(columns={'SegmentoCVM': 'Segmento'}, inplace=True)

#%% [13] Drop NA

df.columns

#df_na = df.dropna()

df = df.dropna()

print(df.head())

#%% [14] Remover setores com menos de 3 papeis:

setores_validos = df.groupby('Setor')['Papel'].nunique()
setores_validos = setores_validos[setores_validos > 2].index  # setores com mais de 2 papéis

# Filtrar apenas setores válidos
df = df[df['Setor'].isin(setores_validos)]


#%% [15] Tabela frequência por Setor

freq_empresas_setor = df[['Papel', 'Setor']].drop_duplicates()
tabela_freq = freq_empresas_setor['Setor'].value_counts().reset_index()
tabela_freq.columns = ['Setor', 'Quantidade de Empresas']
print(tabela_freq)

#%% [16] Tabela frequência por Segmento

freq_empresas_segmento = df[['Papel', 'Segmento']].drop_duplicates()
tabela_freq = freq_empresas_segmento['Segmento'].value_counts().reset_index()
tabela_freq.columns = ['Setor', 'Quantidade de Empresas']
print(tabela_freq)


#%% [17] Estatísticas Descritivas

colunas_float = df.select_dtypes(include='float').columns.tolist()

estatisticas = df[colunas_float].describe().T
estatisticas['Mediana'] = df[colunas_float].median()
estatisticas = estatisticas[['mean', 'Mediana', 'std', 'min', 'max']]
estatisticas.columns = ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo']
estatisticas = estatisticas.round(2)

print(estatisticas)

#%% [18] Transformação de Box-Cox

# Para o cálculo do lambda de Box-Cox
from scipy.stats import boxcox

# 'yast' é uma variável que traz os valores transformados (Y*)
# 'lmbda' é o lambda de Box-Cox
yast, lmbda = boxcox(df['ValorMercado_bi'])

df['ValorMercado_bc'] = yast

df

print ("lambda de Box-Cox: ", lmbda)


#%% [19] Box Plot do Valor de Mercado

plt.figure(figsize=(8, 5))
sns.boxplot(y=df['ValorMercado_bi'], color='skyblue')
plt.title('Boxplot da Variável Alvo')
plt.ylabel('Valor de Mercado (bilhões de R$)')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%% [20] Box Plot do Valor de Mercado após transformação de Box Cox

plt.figure(figsize=(7, 5))
sns.boxplot(y=df['ValorMercado_bc'], color='skyblue')
plt.title('Boxplot da Variável Alvo')
plt.ylabel('Valor de Mercado (após transformação de Box-Cox)')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#%% [21] Valor de Mercado Gráfico

df_ordenado = df.sort_values("ValorMercado_bi", ascending=True)

plt.figure(figsize=(12,6))
plt.bar(df_ordenado["Papel"], df_ordenado["ValorMercado_bi"])

plt.xticks(rotation=90, fontsize=6)
plt.xlabel("Companhias")
plt.ylabel("Valor de Mercado (bi)")
plt.title("Valores de Mercado das Companhias")
plt.tight_layout()
plt.show()

#%% [22] Valor de Mercado Gráfico Vertical

top_n = 80
df_ordenado = df.sort_values("ValorMercado_bi", ascending=False).head(top_n).sort_values("ValorMercado_bi", ascending=True)

plt.figure(figsize=(6,8))
plt.barh(df_ordenado["Papel"], df_ordenado["ValorMercado_bi"])

plt.xticks(fontsize=8)
plt.yticks(fontsize=5)
plt.xlabel("Valor de mercado (bi)", fontsize=10)
plt.ylabel("Companhias", fontsize=10)
plt.title(f"Valor de mercado ({top_n} maiores companhias) estudadas", fontsize=11)
plt.margins(x=0.01, y=0.01)
#plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



#%% [23] Valor de mercado médio por setor

valordemercado_medio = df.groupby('Setor')['ValorMercado_bi'].mean().reset_index()
valordemercado_medio

#%% [24] Valor de mercado médio por setor

valordemercado_medio = df.groupby('Setor')['ValorMercado_bi'].mean().round(2).reset_index()
valordemercado_medio

#In[1.3]: Gráfico do valor de mercado médio das empresas por setor

from matplotlib.lines import Line2D

def quebra_proxima_do_meio(texto):
    palavras = texto.split()
    if len(palavras) <= 1:
        return texto  # Não quebra se for apenas uma palavra
    
    meio = len(palavras) // 2
    # Junta a primeira metade e a segunda metade com quebra de linha
    return " ".join(palavras[:meio]) + "\n" + " ".join(palavras[meio:])

# Gerar labels quebrados no meio
labels_quebrados = [quebra_proxima_do_meio(label) for label in valordemercado_medio['Setor']]

plt.figure(figsize=(15,10))
plt.plot(valordemercado_medio['Setor'], valordemercado_medio['ValorMercado_bi'],
         linewidth=5, color='indigo')
plt.scatter(df['Setor'], df['ValorMercado_bi'],
            alpha=0.5, color='orange', s = 150)
plt.xlabel('Setor $j$ (nível 2)', fontsize=20)
plt.ylabel('Valor de mercado (bilhões de reais)', fontsize=20)
plt.xticks(valordemercado_medio['Setor'], labels_quebrados, fontsize=17, rotation=60)
plt.yticks(fontsize=17)

# Criar legenda manual
legend_elements = [
    Line2D([0], [0], color='indigo', lw=4, label='Média'),                  # Linha azul
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
           markersize=10, alpha=0.7, label='Observações')                   # Bolinhas amarelas
]

plt.legend(handles=legend_elements, fontsize=15, loc='upper left')

plt.show()


#%% [25] Valor de mercado médio por setor (Transformada por Box-Cox)

valordemercado_medio = df.groupby('Setor')['ValorMercado_bc'].mean().round(2).reset_index()
valordemercado_medio

#In[1.3]: Gráfico do valor de mercado médio das empresas por setor

from matplotlib.lines import Line2D

def quebra_proxima_do_meio(texto):
    palavras = texto.split()
    if len(palavras) <= 1:
        return texto  # Não quebra se for apenas uma palavra
    
    meio = len(palavras) // 2
    # Junta a primeira metade e a segunda metade com quebra de linha
    return " ".join(palavras[:meio]) + "\n" + " ".join(palavras[meio:])

# Gerar labels quebrados no meio
labels_quebrados = [quebra_proxima_do_meio(label) for label in valordemercado_medio['Setor']]

plt.figure(figsize=(15,10))
plt.plot(valordemercado_medio['Setor'], valordemercado_medio['ValorMercado_bc'],
         linewidth=5, color='indigo')
plt.scatter(df['Setor'], df['ValorMercado_bc'],
            alpha=0.5, color='orange', s = 150)
plt.xlabel('Setor $j$ (nível 2)', fontsize=20)
plt.ylabel('Valor de mercado (BC)', fontsize=20)
plt.xticks(valordemercado_medio['Setor'], labels_quebrados, fontsize=17, rotation=60)
plt.yticks(fontsize=17)

# Criar legenda manual
legend_elements = [
    Line2D([0], [0], color='indigo', lw=4, label='Média'),                  # Linha azul
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
           markersize=10, alpha=0.7, label='Observações')                   # Bolinhas amarelas
]

plt.legend(handles=legend_elements, fontsize=15, loc='upper left')

plt.show()


#%% [26] Kernel density estimation (KDE) - função densidade de probabilidade
#da variável dependente ('ValorMercado_bi'), com histograma

plt.figure(figsize=(15,10))
sns.histplot(data=df['ValorMercado_bi'], kde=True,
             bins=30, color='deepskyblue')
plt.xlabel('Valor de mercado (bi)', fontsize=20)
plt.ylabel('Contagem', fontsize=20)
plt.tick_params(axis='y', labelsize=17)
plt.tick_params(axis='x', labelsize=17)
plt.show()

#%% [27]: Kernel density estimation (KDE) - função densidade de probabilidade
#da variável dependente transformada ('ValorMercado_bc'), com histograma

plt.figure(figsize=(15,10))
sns.histplot(data=df['ValorMercado_bc'], kde=True,
             bins=30, color='deepskyblue')
plt.xlabel('Valor de mercado após transformação de Box-Cox', fontsize=20)
plt.ylabel('Contagem', fontsize=20)
plt.tick_params(axis='y', labelsize=17)
plt.tick_params(axis='x', labelsize=17)
plt.show()

#%% [28] Boxplot da variável dependente ('valor de mercado') por setor

def quebra_proxima_do_meio(texto):
    palavras = texto.split()
    if len(palavras) <= 1:
        return texto  # Não quebra se for apenas uma palavra
    
    meio = len(palavras) // 2
    return " ".join(palavras[:meio]) + "\n" + " ".join(palavras[meio:])

plt.figure(figsize=(15,10))
sns.boxplot(data=df, x='Setor', y='ValorMercado_bi',
            linewidth=2, orient='v', palette='viridis')
#sns.stripplot(data=df, x='Setor', y='Valordemercado_mi',
#              palette='viridis', jitter=0.2, size=8, alpha=0.5)
plt.ylabel('Valor de mercado (bi)', fontsize=20)
plt.xlabel('Setor $j$ (nível 2)', fontsize=20)
plt.tick_params(axis='y', labelsize=17)
plt.tick_params(axis='x', labelsize=17)

labels_quebrados = [quebra_proxima_do_meio(label.get_text()) for label in plt.gca().get_xticklabels()]
plt.xticks(range(len(labels_quebrados)), labels_quebrados, fontsize=17, rotation=60)

plt.show()

#%% [29] Boxplot da variável dependente ('valor de mercado') por setor
# após transformação de Box-Cox

def quebra_proxima_do_meio(texto):
    palavras = texto.split()
    if len(palavras) <= 1:
        return texto  # Não quebra se for apenas uma palavra
    
    meio = len(palavras) // 2
    return " ".join(palavras[:meio]) + "\n" + " ".join(palavras[meio:])

plt.figure(figsize=(15,10))
sns.boxplot(data=df, x='Setor', y='ValorMercado_bc',
            linewidth=2, orient='v', palette='viridis')
#sns.stripplot(data=df, x='Setor', y='Valordemercado_mi',
#              palette='viridis', jitter=0.2, size=8, alpha=0.5)
plt.ylabel('Valor de mercado (transformado por Box-Cox)', fontsize=20)
plt.xlabel('Setor $j$ (nível 2)', fontsize=20)
plt.tick_params(axis='y', labelsize=17)
plt.tick_params(axis='x', labelsize=17)

labels_quebrados = [quebra_proxima_do_meio(label.get_text()) for label in plt.gca().get_xticklabels()]
plt.xticks(range(len(labels_quebrados)), labels_quebrados, fontsize=17, rotation=60)

plt.show()

#%% [30] Kernel density estimation (KDE) - função densidade de probabilidade
#da variável dependente ('desempenho'), com histograma e por escola separadamente
#(função 'GridSpec' do pacote 'matplotlib.gridspec')

setores = df['Setor'].unique()

fig = plt.figure(figsize=(15, 14))
gs = GridSpec(len(setores) // 2 + 1, 2, figure=fig)

for i, setor in enumerate(setores):
    ax = fig.add_subplot(gs[i])

    # Subset dos dados por escola
    df_setor = df[df['Setor'] == setor]

    # Densidade dos dados
    densidade = gaussian_kde(df_setor['ValorMercado_bi'])
    x_vals = np.linspace(min(df_setor['ValorMercado_bi']),
                         max(df_setor['ValorMercado_bi']), len(df_setor))
    y_vals = densidade(x_vals)

    # Plotagem da density area
    ax.fill_between(x_vals, y_vals,
                    color=sns.color_palette('viridis',
                                            as_cmap=True)(i/len(setores)),
                    alpha=0.3)
    
    # Adiciona o histograma
    sns.histplot(df_setor['ValorMercado_bi'], ax=ax, stat="density", color="black",
                 edgecolor="black", fill=True, 
                 bins=15, alpha=0.1)
    ax.set_title(f'{setor}', fontsize=15)
    ax.set_ylabel('Densidade')
    ax.set_xlabel('Valor de Mercado (bi)')

plt.tight_layout()
plt.show()


#%% [31] Kernel density estimation (KDE) - função densidade de probabilidade
#da variável dependente ('Valor de Mercado'), com histograma e por setor separadamente
#(função 'GridSpec' do pacote 'matplotlib.gridspec')
# Transformada por Box-Cox

from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec

setores = df['Setor'].unique()

fig = plt.figure(figsize=(15, 14))
gs = GridSpec(len(setores) // 2 + 1, 2, figure=fig)

for i, setor in enumerate(setores):
    ax = fig.add_subplot(gs[i])

    # Subset dos dados por escola
    df_setor = df[df['Setor'] == setor]

    # Densidade dos dados
    densidade = gaussian_kde(df_setor['ValorMercado_bc'])
    x_vals = np.linspace(min(df_setor['ValorMercado_bc']),
                         max(df_setor['ValorMercado_bc']), len(df_setor))
    y_vals = densidade(x_vals)

    # Plotagem da density area
    ax.fill_between(x_vals, y_vals,
                    color=sns.color_palette('viridis',
                                            as_cmap=True)(i/len(setores)),
                    alpha=0.3)
    
    # Adiciona o histograma
    sns.histplot(df_setor['ValorMercado_bc'], ax=ax, stat="density", color="black",
                 edgecolor="black", fill=True, 
                 bins=15, alpha=0.1)
    ax.set_title(f'{setor}', fontsize=15)
    ax.set_ylabel('Densidade')
    ax.set_xlabel('Valor de Mercado (Transformado por Box-Cox)')

plt.tight_layout()
plt.show()


#%% [32]

df.columns

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


colunas = ['PL', 'PSR', 'PVP', 'EVEBITDA', 'EVEBIT', 'PEBITDA', 'PEBIT', 'PAtivo',
           'PCapGiro', 'PAtCircLiq', 'VPA', 'LPA', 'GiroAtivos', 'ROE', 'ROIC',
           'ROA', 'DivLiqEBITDA', 'DivLiqEBIT', 'PatrAtiv', 'PassAtiv','LiqCorr']

corr = df[colunas].corr()

# Gerar o mapa de calor
plt.figure(figsize=(20, 16), dpi=300)
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    annot_kws={"size": 10},        # Tamanho dos números nas células
    xticklabels=1, yticklabels=1  # Mostra todos os rótulos
)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Matriz de Correlação das variáveis explicativas", fontsize=20)
plt.tight_layout()
plt.show()

df.columns

#%% [33] FÓRMULA PRO MODELO

df.columns

#df = df.dropna()

lista_colunas = ['PL', 'PSR', 'PVP', 'EVEBITDA', 'EVEBIT', 'PEBITDA', 'PEBIT',
       'PAtivo', 'PCapGiro', 'PAtCircLiq', 'VPA', 'LPA', 'GiroAtivos', 'ROE',
       'ROIC', 'ROA', 'DivLiqEBITDA', 'DivLiqEBIT', 'PatrAtiv', 'PassAtiv',
       'LiqCorr']

formula_modelo = ' + '.join(lista_colunas)
#formula_modelo = "ValorMercado ~ " + formula_modelo
print(formula_modelo)


# In[34]: Estimação do 'modelo_ols'

modelo_ols = sm.OLS.from_formula("ValorMercado_bi ~ " + formula_modelo, df).fit()

# Parâmetros do 'modelo_ols_dummies'
modelo_ols.summary()

#%% [35] Procedimento Stepwise

from statstests.process import stepwise

# Estimação do modelo por meio do procedimento Stepwise

modelo_ols_step = stepwise(modelo_ols, pvalue_limit=0.05)

modelo_ols_step.summary()

# In[36]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Wilk (n < 30)
# from scipy.stats import shapiro
# shapiro(modelo_linear.resid)

# Teste de Shapiro-Francia (n >= 30)
# Carregamento da função 'shapiro_francia' do pacote 'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

from statstests.tests import shapiro_francia

# Teste de Shapiro-Francia: interpretação
teste_sf = shapiro_francia(modelo_ols_step.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')


# In[37]: Histograma dos resíduos do modelo OLS linear

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
hist1 = sns.histplot(data=modelo_ols_step.resid, kde=True, bins=25,
                     color = 'darkorange', alpha=0.4, edgecolor='silver',
                     line_kws={'linewidth': 3})
hist1.get_lines()[0].set_color('orangered')
plt.xlabel('Resíduos', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


# In[38]: Estimando um novo modelo OLS com variável dependente
#transformada por Box-Cox

modelo_bc = sm.OLS.from_formula("ValorMercado_bc ~ " + formula_modelo, df).fit()

# Parâmetros do 'modelo_bc'
modelo_bc.summary()

#%% [39] Modelo BC Step

modelo_bc_step = stepwise(modelo_bc, pvalue_limit=0.05)

modelo_bc_step.summary()

#%% [40] Teste de Shapiro-Francia:
    
from statstests.tests import shapiro_francia

teste_sf = shapiro_francia(modelo_bc_step.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')
    
    
# In[41]: Histograma dos resíduos do modelo OLS linear

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))
hist1 = sns.histplot(data=modelo_bc_step.resid, kde=True, bins=25,
                     color = 'darkorange', alpha=0.4, edgecolor='silver',
                     line_kws={'linewidth': 3})
hist1.get_lines()[0].set_color('orangered')
plt.xlabel('Resíduos', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()


#%% [42]

df['yhat_modelo_bc'] = (modelo_bc_step.fittedvalues * lmbda + 1) ** (1 / lmbda)
df

df_filtrado = df[["ValorMercado_bi", "yhat_modelo_bc"]]


# In[43]: Gráfico para a comparação dos fitted values do modelo OLS

df['fitted_OLS'] = modelo_ols_step.fittedvalues
df_fitted = df[['ValorMercado_bi','fitted_OLS']]

plt.figure(figsize=(15,10))
sns.regplot(x=df['ValorMercado_bi'],
            y=df['ValorMercado_bi'],
            ci=None, marker='o', order=1,
            scatter_kws={'color':'black', 's':50, 'alpha':0.5},
            line_kws={'color':'black', 'linewidth':2, 'linestyle':'--'}
            )
sns.regplot(x=df['ValorMercado_bi'],
             y=df['yhat_modelo_bc'],
             ci=None, marker='o', order=1,
             scatter_kws={'color':'navy', 's':50, 'alpha':0.5},
             line_kws={'color':'navy', 'linewidth':1,
                       'label':'OLS BC'})
# sns.regplot(x=df['ValorMercado_bi'],
#             y=modelo_ols_step.fittedvalues,
#             ci=None, marker='s', order=1,
#             scatter_kws={'color':'deeppink', 's':50, 'alpha':0.5},
#             line_kws={'color':'deeppink', 'linewidth':1,
#                       'label':'OLS'})
plt.xlabel('Valor de Mercado', fontsize=20)
plt.ylabel('Fitted Values', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(fontsize=20)
plt.show()

#%% [44] MULTINIVEL
# Modelo Nulo HLM2

modelo_nulo_hlm2 = sm.MixedLM.from_formula(formula='ValorMercado_bc ~ 1',
                                           groups='Setor',
                                           re_formula='1',
                                           data=df).fit()

modelo_nulo_hlm2.summary()

#%% [45]
#ICC Modelo Nulo:
# Coeficiente de Setor Var / (Coeficiente de Setor Var + Scale)

0.639 / (0.639 + 3.0920)

#Coef. de Setor Var / Std Err. de Setor Var = Abcissa da distribuição normal padronizada
# Acima de 2 (significante) - LogLik acima de um modelo nulo OLS

0.639  /    0.238


# In[1.14]: Análise da significância estatística dos efeitos aleatórios de
#intercepto

teste = float(modelo_nulo_hlm2.cov_re.iloc[0, 0]) /\
    float(pd.DataFrame(modelo_nulo_hlm2.summary().tables[1]).iloc[1, 1])

p_value = 2 * (1 - stats.norm.cdf(abs(teste)))

print(f"Estatística z para a Significância dos Efeitos Aleatórios: {teste:.3f}")
print(f"P-valor: {p_value:.3f}")

if p_value >= 0.05:
    print("Ausência de significância estatística dos efeitos aleatórios ao nível de confiança de 95%.")
else:
    print("Efeitos aleatórios contextuais significantes ao nível de confiança de 95%.")
    
    
# In[1.19]:
#     ESTIMAÇÃO DO MODELO COM INTERCEPTOS E INCLINAÇÕES ALEATÓRIOS HLM2

# Rodei com cada uma das variáveis explicativas do OLS.

modelo_intercept_inclin_hlm2 = sm.MixedLM.from_formula(formula='ValorMercado_bc ~ ROIC',
                                                       groups='Setor',
                                                       re_formula= 'ROIC',
                                                       data=df).fit()

# Parâmetros do 'modelo_intercept_inclin_hlm2'
modelo_intercept_inclin_hlm2.summary()

# In[1.20]: Análise da significância estatística dos efeitos aleatórios de
#intercepto

teste = float(modelo_intercept_inclin_hlm2.cov_re.iloc[0, 0]) /\
    float(pd.DataFrame(modelo_intercept_inclin_hlm2.summary().tables[1]).iloc[2, 1])  #Setor var Coef/Setor Var Std Err

p_value = 2 * (1 - stats.norm.cdf(abs(teste)))

print(f"Estatística z para a Significância dos Efeitos Aleatórios de intercepto: {teste:.3f}")
print(f"P-valor dos efeitos aleatórios de intercepto: {p_value:.3f}")

if p_value >= 0.05:
    print("Ausência de significância estatística dos efeitos aleatórios de intercepto ao nível de confiança de 95%.")
else:
    print("Efeitos aleatórios de intercepto contextuais significantes ao nível de confiança de 95%.")

# In[1.21]: Análise da significância estatística dos efeitos aleatórios de
#inclinação

teste = float(modelo_intercept_inclin_hlm2.cov_re.iloc[1, 1]) /\
    float(pd.DataFrame(modelo_intercept_inclin_hlm2.summary().tables[1]).iloc[4, 1]) #PSR Var / PSR Var Std Err

p_value = 2 * (1 - stats.norm.cdf(abs(teste)))

print(f"Estatística z para a Significância dos Efeitos Aleatórios de inclinação: {teste:.3f}")
print(f"P-valor dos efeitos aleatórios de inclinação: {p_value:.3f}")

if p_value >= 0.05:
    print("Ausência de significância estatística dos efeitos aleatórios de inclinação ao nível de confiança de 95%.")
else:
    print("Efeitos aleatórios de inclinação  contextuais significantes ao nível de confiança de 95%.")


#%%
#  ESTIMAÇÃO DO MODELO FINAL COM INTERCEPTOS E INCLINAÇÕES ALEATÓRIOS HLM2   #
#  e inserção da variável de setor.

# Estimação do modelo final com interceptos e inclinações aleatórios
# Com 4 variáveis
# Não usei esse

modelo_final_hlm2 = sm.MixedLM.from_formula(formula='ValorMercado_bc ~ PVP + PAtivo + LPA + ROIC + MargBrutSetor +\
                                            + PVP:MargBrutSetor + PAtivo:MargBrutSetor \
                                            + LPA:MargBrutSetor + ROIC:MargBrutSetor',
                                            groups='Setor',
                                            re_formula='1',
                                            data=df).fit()

# Parâmetros do modelo 'modelo_final_hlm2'
modelo_final_hlm2.summary()

#%% Com 1 variável: PVP

modelo_final_hlm2_pvp = sm.MixedLM.from_formula(formula='ValorMercado_bc ~ PVP + MargBrutSetor +\
                                            + PVP:MargBrutSetor',
                                            groups='Setor',
                                            re_formula='1',
                                            data=df).fit()

# Parâmetros do modelo 'modelo_final_hlm2'
modelo_final_hlm2_pvp.summary()

#%% Com 1 variável: ROIC

modelo_final_hlm2_roic = sm.MixedLM.from_formula(formula='ValorMercado_bc ~ ROIC + MargBrutSetor +\
                                            + ROIC:MargBrutSetor',
                                            groups='Setor',
                                            re_formula='1',
                                            data=df).fit()

# Parâmetros do modelo 'modelo_final_hlm2'
modelo_final_hlm2_roic.summary()


#%% Só com variáveis de nível 1
#Não está no trabalho (incluir?)

modelo_final_hlm2 = sm.MixedLM.from_formula(formula='ValorMercado_bc ~ PAtivo + LPA + ROIC',
                                            groups='Setor',
                                            re_formula='1',
                                            data=df).fit()

# Parâmetros do modelo 'modelo_final_hlm2'
modelo_final_hlm2.summary()


# In[1.40]:
#                COMPARAÇÃO COM UM MODELO OLS COM DUMMIES                    #

# Dummizando a variável 'setor'. (n-1 dummies)

df_setor_dummies = pd.get_dummies(df, columns=['Setor'],
                                         dtype=int,
                                         drop_first=True)

df_setor_dummies.head(10)

#%% Ajustando nomes de colunas

df_setor_dummies.columns

mapa_colunas = {
    'Setor_Comunicações': 'SetorCom',
    'Setor_Consumo Cíclico': 'Setor_ConCicl',
    'Setor_Consumo não Cíclico': 'Setor_ConNCicl',
    'Setor_Financeiro': 'Setor_Fin',
    'Setor_Materiais Básicos': 'Setor_MatBas',
    'Setor_Petróleo, Gás e Biocombustíveis': 'Setor_OleoGasBio',
    'Setor_Saúde': 'Setor_Saude',
    'Setor_Tecnologia da Informação': 'Setor_TecInfo',
    'Setor_Utilidade Pública': 'Setor_UtilPub',
}

df_setor_dummies = df_setor_dummies.rename(columns=mapa_colunas)

print(df_setor_dummies.columns)

#%%
df_setor_dummies.columns

#df = df.dropna()

lista_colunas = ['PL', 'PSR', 'PVP', 'EVEBITDA', 'EVEBIT', 'PEBITDA', 'PEBIT',
       'PAtivo', 'PCapGiro', 'PAtCircLiq', 'VPA', 'LPA', 'GiroAtivos', 'ROE',
       'ROIC', 'ROA', 'DivLiqEBITDA', 'DivLiqEBIT', 'PatrAtiv', 'PassAtiv',
       'LiqCorr', 'SetorCom', 'Setor_ConCicl', 'Setor_ConNCicl',
       'Setor_Fin', 'Setor_MatBas', 'Setor_OleoGasBio', 'Setor_Saude',
       'Setor_TecInfo', 'Setor_UtilPub' ]

formula_modelo = ' + '.join(lista_colunas)
#formula_modelo = "ValorMercado ~ " + formula_modelo
print(formula_modelo)

# In[1.42]: Estimação do modelo com n-1 dummies propriamente dito

modelo_ols_dummies = sm.OLS.from_formula("ValorMercado_bc ~ " + formula_modelo, df_setor_dummies).fit()

# Parâmetros do 'modelo_ols_dummies'
modelo_ols_dummies.summary()

# In[1.43]: Procedimento Stepwise para o 'modelo_ols_dummies'

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

from statstests.process import stepwise

# Estimação do modelo por meio do procedimento Stepwise

modelo_ols_dummies_step = stepwise(modelo_ols_dummies, pvalue_limit=0.05)


#%% Gráfico para comparação visual dos logLiks de todos os modelos estimados

df_llf = pd.DataFrame({'modelo':['OLS Stepwise',
                                 'HLM2 Nulo',
                                 'HLM2 com Int. e Incl. Aleat.',
                                 'HLM2 com X=ROIC',
                                 'HLM2 com X=PVP',
                                 'OLS com dummies e Stepwise'],
                      'loglik':[modelo_bc_step.llf,
                                modelo_nulo_hlm2.llf,
                                modelo_intercept_inclin_hlm2.llf,
                                modelo_final_hlm2_roic.llf,
                                modelo_final_hlm2_pvp.llf,
                                modelo_ols_dummies_step.llf]})

df_llf = df_llf.sort_values(by="loglik", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(15,15))

c = ['dimgray','darkslategray','navy','dodgerblue','indigo','deeppink']

ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
ax.bar_label(ax1, label_type='center', color='white', fontsize=40)
ax.set_ylabel("Modelo Proposto", fontsize=24)
ax.set_xlabel("LogLik", fontsize=24)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=20)
plt.show()

