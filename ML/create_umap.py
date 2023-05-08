import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from common.rdkitdescriptors import mol2feature, TransferSmiles
import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA

def check(data):
    if "Unnamed: 0" in data.columns:
        data.drop("Unnamed: 0", axis=1, inplace=True)
    return data


def append(smiles, data):
    mol = TransferSmiles(smiles)
    extra = mol2feature(mol)
    extra["category"] = 1 # dpphを別色で使いたければここの数字を変えて
    extra.index = [smiles]
    if "category" not in data.columns:
        data["category"] = 0
    added = data.append(extra)
    return added


def scale(data):
    # データを標準化して返す
    # csvを使う
    smiles = data.index
    before_sc = data.reset_index(drop=True)
    col = data.columns
    sc = StandardScaler()
    after_sc = check(pd.DataFrame(sc.fit_transform(before_sc), columns=col, index=smiles))
    return after_sc

def fit(data):
    # 化合物データをumapで二次元に削減
    data.reset_index(drop=True, inplace=True)
    reducer = umap.UMAP(n_neighbors=15, random_state=41)
    reducer = reducer.fit(data)
    return reducer


def transform(data, reducer):
    smiles = data.index
    data.reset_index(drop=True, inplace=True)
    vecs = check(pd.DataFrame(reducer.transform(data), columns=['axis 1', 'axis 2'], index=smiles))
    return vecs


def show(data):
    # 図を出力
    fig = plt.figure()
    for i in range(data["category"].nunique()):
        plt.scatter(data[data["category"] == i]["axis 1"], data[data["category"] == i]["axis 2"], label=i)
    plt.title("UMAP plot")
    plt.xlabel("axis 1")
    plt.ylabel("axis 2")
    plt.legend()
    return fig

def show_plotly(data):
    fig = go.Figure()
    """fig.add_traces(go.Scatter(x = data[data['category'] == i]['axis 1'],
                              y = data[data['category'] == i]['axis 2'],
                              ))"""
    for i in range(data['category'].nunique()):
        fig.add_traces(go.Scatter(x = data[data['category'] == i]['axis 1'],
                                  y = data[data['category'] == i]['axis 2'],
                                  text = data['SMILES']
                                  ))
    return fig


def pca_fit(data):
    pca = PCA(n_components = 2)
    df_pca = pd.DataFrame(pca.fit_transform(data))
    df_pca = df_pca.rename(columns = {0: 'axis 1', 1: 'axis 2'})
    return df_pca