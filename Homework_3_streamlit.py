import streamlit
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

streamlit.set_page_config(page_title="Homework 3", page_icon=":bar_chart:", layout="wide")

streamlit.title("Homework 3: Data Visualization")
streamlit.header("Exploratory Data Analysis and Visualization of the World Happiness Report Dataset")

#Reading the data 
path = '/Users/halaassaf/Desktop/Fall 2026/Data Viz/Class 3/Data/'
data_2015 = pd.read_csv(path + '2015.csv')
data_2016 = pd.read_csv(path + '2016.csv')
data_2017 = pd.read_csv(path + '2017.csv')
data_2018 = pd.read_csv(path + '2018.csv')
data_2019 = pd.read_csv(path + '2019.csv')

#Updating the mapping
data_2015 = data_2015.rename(columns={
    'Happiness Rank': 'Rank',
    'Happiness Score': 'Score',
    'Economy (GDP per Capita)': 'GDP',
    'Family': 'Social_Support',
    'Health (Life Expectancy)': 'Life_Expectancy',
    'Trust (Government Corruption)': 'Trust',
    'Dystopia Residual': 'Dystopia_Residual',
})

data_2016 = data_2016.rename(columns={
    'Happiness Rank': 'Rank',
    'Happiness Score': 'Score',
    'Economy (GDP per Capita)': 'GDP',
    'Family': 'Social_Support',
    'Health (Life Expectancy)': 'Life_Expectancy',
    'Trust (Government Corruption)': 'Trust',
    'Dystopia Residual': 'Dystopia_Residual',
})

data_2017 = data_2017.rename(columns={
    'Happiness.Rank': 'Rank',
    'Happiness.Score': 'Score',
    'Economy..GDP.per.Capita.': 'GDP',
    'Family': 'Social_Support',
    'Health..Life.Expectancy.': 'Life_Expectancy',
    'Trust..Government.Corruption.': 'Trust',
    'Dystopia.Residual': 'Dystopia_Residual',
})

data_2018 = data_2018.rename(columns={
    'Overall rank': 'Rank',
    'Country or region': 'Country',
    'Score': 'Score',
    'GDP per capita': 'GDP',
    'Social support': 'Social_Support',
    'Healthy life expectancy': 'Life_Expectancy',
    'Freedom to make life choices': 'Freedom',
    'Perceptions of corruption': 'Trust',
})

data_2019 = data_2019.rename(columns={
    'Overall rank': 'Rank',
    'Country or region': 'Country',
    'Score': 'Score',
    'GDP per capita': 'GDP',
    'Social support': 'Social_Support',
    'Healthy life expectancy': 'Life_Expectancy',
    'Freedom to make life choices': 'Freedom',
    'Perceptions of corruption': 'Trust',
})

#Data Processing 
dfs = [data_2015, data_2016, data_2017, data_2018, data_2019]
years = [2015, 2016, 2017, 2018, 2019]

for df, year in zip(dfs, years):
    df['Year'] = year

concat_data = pd.concat(dfs, ignore_index=True)

concat_data_clean = concat_data.drop(columns=[
    'Standard Error',
    'Lower Confidence Interval',
    'Upper Confidence Interval',
    'Whisker.high',
    'Whisker.low'
])

streamlit.subheader("Data Overview")
streamlit.write(concat_data_clean.head())

streamlit.subheader("Missing Values by Column")
streamlit.write(concat_data.isna().sum())

numeric_df = concat_data_clean.select_dtypes(include='number')
corr_matrix = numeric_df.corr()

sorted_cols = corr_matrix['Score'].sort_values(ascending=False).index
corr_sorted = corr_matrix.loc[sorted_cols, sorted_cols]

streamlit.subheader("Correlation Heatmap")
fig1, ax1 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_sorted, annot=True, cmap='coolwarm', fmt='.2f', ax=ax1)
ax1.set_title('Correlation Heatmap: World Happiness Factors')
streamlit.pyplot(fig1)

streamlit.markdown("""
GDP, life expectancy, and social support show the strongest correlations with
happiness score, each in the range of 0.7 to 0.8. Generosity shows almost no
correlation with score, and trust is similarly weak. Economic and health
indicators appear to be more closely associated with happiness than generosity
in this dataset.
""")

streamlit.subheader("Score vs GDP")
pair = ("Score", "GDP")
r = numeric_df[list(pair)].corr().iloc[0, 1]
fig2, ax2 = plt.subplots(figsize=(6.5, 4))
ax2.scatter(numeric_df[pair[0]], numeric_df[pair[1]], alpha=0.6, color="#2E6E8E")
ax2.set_xlabel(pair[0])
ax2.set_ylabel(pair[1])
ax2.set_title(f"{pair[0]} vs {pair[1]} (r = {r:.2f})")
streamlit.pyplot(fig2)

streamlit.markdown("""
Happiness score increases consistently with GDP. The relationship is not
perfectly linear, but countries with higher GDP tend to report higher
happiness scores on average.
""")

streamlit.subheader("Score vs Life Expectancy")
pair = ("Score", "Life_Expectancy")
r = numeric_df[list(pair)].corr().iloc[0, 1]
fig3, ax3 = plt.subplots(figsize=(6.5, 4))
ax3.scatter(numeric_df[pair[0]], numeric_df[pair[1]], alpha=0.6, color="#2E6E8E")
ax3.set_xlabel(pair[0])
ax3.set_ylabel(pair[1])
ax3.set_title(f"{pair[0]} vs {pair[1]} (r = {r:.2f})")
streamlit.pyplot(fig3)

streamlit.markdown("""
Life expectancy shows a similar pattern to GDP, with longer life expectancy
associated with higher happiness scores. This is consistent with the fact that
health and economic outcomes tend to be correlated across countries.
""")

streamlit.subheader("Score vs Social Support")
pair = ("Score", "Social_Support")
r = numeric_df[list(pair)].corr().iloc[0, 1]
fig4, ax4 = plt.subplots(figsize=(6.5, 4))
ax4.scatter(numeric_df[pair[0]], numeric_df[pair[1]], alpha=0.6, color="#2E6E8E")
ax4.set_xlabel(pair[0])
ax4.set_ylabel(pair[1])
ax4.set_title(f"{pair[0]} vs {pair[1]} (r = {r:.2f})")
streamlit.pyplot(fig4)

streamlit.markdown("""
Social support follows the same general upward trend, though the data points
are more dispersed than in the GDP or life expectancy plots. This suggests a
positive but comparatively weaker relationship with happiness score.
""")

streamlit.header("PCA: Reducing 6 Features to Their Main Components")

features = ['GDP', 'Social_Support', 'Life_Expectancy', 'Freedom', 'Trust', 'Generosity']
target = 'Score'
numeric_df_clean = numeric_df[features + [target]].dropna()
X = StandardScaler().fit_transform(numeric_df_clean[features])
y = numeric_df_clean[target]

pca2 = PCA(n_components=2, svd_solver='full')
pcs2 = pca2.fit_transform(X)

loadings = pd.DataFrame(
    pca2.components_.T,
    columns=[f'PC{i+1}' for i in range(pca2.n_components_)],
    index=features
)

streamlit.subheader("PCA Loadings (2 components)")
streamlit.dataframe(loadings.round(3).sort_values(by='PC1', key=abs, ascending=False))

streamlit.markdown("""
PC1 can be interpreted as an economic and health well-being axis, as GDP, life
expectancy, and social support load heavily and in the same direction. PC2
reflects a different pattern, contrasting generosity and trust against GDP and
life expectancy. Since happiness score was excluded from the PCA input, this
structure is derived entirely from the relationships among the remaining six
features.
""")

pca3 = PCA(n_components=3, svd_solver='full')
pcs3 = pca3.fit_transform(X)

streamlit.subheader("Explained Variance (3 components)")
streamlit.write(pca3.explained_variance_ratio_.round(3))
streamlit.write(f"Cumulative: {pca3.explained_variance_ratio_.sum():.1%}")

streamlit.markdown("""
The first three principal components together account for approximately 80% of
the total variance, with the first component alone explaining roughly half.
Additional components contribute progressively smaller amounts of explained
variance.
""")

streamlit.subheader("Happiness Score Across PC1 & PC2")
fig5, ax5 = plt.subplots(figsize=(7, 4.5))
sc = ax5.scatter(pcs2[:, 0], pcs2[:, 1], c=y, cmap="viridis", s=30)
ax5.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.0%})")
ax5.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.0%})")
ax5.set_title("Happiness Score across PC1 & PC2")
fig5.colorbar(sc, ax=ax5, label="Happiness Score")
streamlit.pyplot(fig5)

streamlit.markdown("""
Although happiness score was not used to construct these components, PC1 shows
a clear gradient with score, with lower scores concentrated on the left and
higher scores on the right. PC2 shows little visible relationship with score,
suggesting that the dimension it represents, primarily generosity and trust,
contributes little to explaining happiness on its own.
""")

corr_pc1 = np.corrcoef(pcs2[:, 0], y)[0, 1]
corr_pc2 = np.corrcoef(pcs2[:, 1], y)[0, 1]

streamlit.subheader("Component Correlations with Score")
streamlit.write(f"Correlation PC1 vs Score: {corr_pc1:.3f}")
streamlit.write(f"Correlation PC2 vs Score: {corr_pc2:.3f}")

streamlit.subheader("Raw Feature Correlations with Score")
correlations = numeric_df_clean[features].corrwith(y).sort_values(ascending=False)
streamlit.write(correlations.round(3))

streamlit.markdown("""
These correlations confirm the visual pattern: PC1 correlates with happiness
score at 0.87, while PC2 shows a negligible correlation of -0.07. The
correlations between the individual features and score support the same
conclusion: GDP, life expectancy, and social support are the strongest
predictors, freedom and trust show moderate associations, and generosity has
the weakest relationship with happiness score.
""")