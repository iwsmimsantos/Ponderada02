import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - Importe os dados medical_examination.csv e atribua-os à variável df
df = pd.read_csv('medical_examination.csv')

# 2 - Crie a coluna overweight na variável df
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)) > 25
df['overweight'] = df['overweight'].astype(int)

# 3 - Normalize os dados tornando 0 sempre bom e 1 sempre ruim.
# Se o valor de cholesterol ou gluc for 1, defina o valor como 0.
# Se o valor for maior que 1, defina o valor como 1.
df['cholesterol'] = df['cholesterol'].replace({1: 0, 2: 0, 3: 1})
df['gluc'] = df['gluc'].replace({1: 0, 2: 0, 3: 1})

# Função para desenhar o gráfico categórico
def draw_cat_plot():
    # Crie um DataFrame para o gráfico do gato usando pd.melt
    df_cat = pd.melt(df, id_vars='cardio', value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Agrupe e reformate os dados df_cat para dividi-los por cardio. Mostre as contagens de cada recurso.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='count')
    
    # Crie um gráfico que mostre as contagens de valores dos recursos categóricos usando sns.catplot().
    fig = sns.catplot(x='variable', hue='value', col='cardio',
                      data=df_cat, kind='count', height=4, aspect=1)
    
    # Defina os rótulos dos eixos
    fig.set_axis_labels("variable", "total")
    
    return fig.fig

# Função para desenhar o Mapa de Calor
def draw_heat_map():
    # Limpe os dados na variável df_heat filtrando segmentos incorretos
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &  
                 (df['height'] >= df['height'].quantile(0.025)) &  
                 (df['height'] <= df['height'].quantile(0.975)) &  
                 (df['weight'] >= df['weight'].quantile(0.025)) &  
                 (df['weight'] <= df['weight'].quantile(0.975))]
    
    # Calcule a matriz de correlação e armazene-a na variável corr
    corr = df_heat.corr()
    
    # Gere uma máscara para o triângulo superior e armazene-a na variável mask
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Configurar a figura do matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))

    # Trace a matriz de correlação usando sns.heatmap()
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', square=True,
                cbar_kws={"shrink": .8}, ax=ax)

    return fig  

# Não modifique as próximas duas linhas
if __name__ == "__main__":
    draw_cat_plot().savefig('catplot.png')
    draw_heat_map().savefig('heatmap.png')