#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap

# Config
csv_name= 'demographics_cln'
#disease_name= 'AD'
cohort= 'aibl'
model= 'sfcne'
feature_csv = f'/mnt/bulk-neptune/radhika/project/features/{cohort}/{model}/_features.csv'  # Your saved features
metadata_csv = f'/mnt/bulk-neptune/radhika/project/data/{cohort}/{csv_name}.csv'  # Your original metadata

#%%
# Load feature CSV
df_features = pd.read_csv(feature_csv)
print(df_features)
# Load metadata
df_metadata = pd.read_csv(metadata_csv)
print(df_metadata)
#%%
print("Duplicates in features:", df_features['eid'].duplicated().sum())
print("Duplicates in metadata:", df_metadata['eid'].duplicated().sum())
df_features = df_features.drop_duplicates(subset='eid', keep='first')
df_metadata = df_metadata.drop_duplicates(subset='eid', keep='first')
df= pd.merge(df_features, df_metadata, on='eid', how='inner')
print(df)
#%%
# Extract features (all numeric columns except 'eid', 'sex', 'age', 'ad')
feature_cols = [col for col in df.columns if col.isdigit()]

X = df[feature_cols].values
print(X.shape)
#%%
# UMAP
reducer = umap.UMAP(n_neighbors=10, min_dist=1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(X)
print(f"✅ UMAP shape: {embedding}")

# Add embedding to dataframe
df['UMAP-1'] = embedding[:, 0]
df['UMAP-2'] = embedding[:, 1]

#%%
def plot_embedding_with_metadata(
    embedding: np.ndarray,
    metadata: np.ndarray,
    metadata_name: str,
    palette: dict = None,
    point_size: int = 400,
    alpha: float = 0.5,
    figsize: tuple = (10, 5),
    show_legend: bool = True,
    discrete: bool = True,
    bins: list = None,
    legend_fontsize: int = 30,
    legend_title_fontsize: int = 35,
    colorbar_fontsize: int = 10,
    colorbar_labelsize: int = 30, 
    frameon: bool = True,
):
    """
    Plot UMAP with metadata overlay and configurable legend/colorbar font sizes.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not discrete:
        # Continuous: Use colorbar below
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=metadata.astype(float),
            cmap='viridis_r',
            s=point_size,
            alpha=alpha, 
            linewidth=0
        )
        cbar = fig.colorbar(
            scatter,
            ax=ax,
            orientation='vertical',
            pad=0,
        )
        #cbar.set_label(metadata_name, fontsize=colorbar_labelsize)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)

    else:
        # Discrete: Use seaborn scatterplot and legend
        if bins:
            metadata_binned = pd.cut(metadata.astype(float), bins=bins).astype(str)
        else:
            metadata_binned = metadata

        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=metadata_binned,
            palette=palette,
            s=point_size,
            alpha=alpha,
            linewidth=0,
            legend=show_legend,
            ax=ax
        )

        if show_legend:
            ax.legend(
                fontsize=legend_fontsize,
                title_fontsize=legend_title_fontsize,
                loc='upper right',
                frameon=True
            )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{metadata_name}", fontsize=40)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()

#%%#%%
# Load metadata from the same CSV

# Extract metadata columns
sex_array = df['sex'].map({0: 'F', 1: 'M'}).values
age_array = df['age'].values
#split_array = metadata_df['split'].values
#disease_array = metadata_df[column_name].map({0: 'CN', 1: disease_name}).values   # dynamic disease column

# Define plotting configs
plot_configs = [
    {
        'metadata': sex_array,
        'metadata_name': 'Sex',
        'palette': {'F': 'lightsalmon', 'M': 'skyblue'},
        'discrete': True
    },
    {
        'metadata': age_array,
        'metadata_name': 'Age',
        'discrete': False
    },
    #{
    #    'metadata': split_array,
    #    'metadata_name': 'Split',
    #    'palette': {'train': 'olive', 'val': 'darkred', 'test': 'cornflowerblue'},
    #    'discrete': True
    #},
    #{
    #    'metadata': disease_array,
    #    'metadata_name': 'Disease',
    #    'palette': {'CN': '#4682B4', disease_name: '#DC143C'},
    #    'discrete': True
    #}
]


#%%
# Plot each
for cfg in plot_configs:
    metadata = cfg['metadata']
    print(metadata.shape)

    plot_embedding_with_metadata(
        embedding=embedding,
        metadata=metadata,
        metadata_name=cfg['metadata_name'],
        palette=cfg.get('palette', None),
        discrete=cfg['discrete'],
        point_size=100,
        alpha=0.7,
        legend_fontsize=30,
        frameon=True,
        legend_title_fontsize=35,
        colorbar_fontsize=30,
        colorbar_labelsize=30
    )
# %%
