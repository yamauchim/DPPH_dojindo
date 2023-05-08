import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors as md
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
from common.rdkitdescriptors import MakeECFP
import pickle

descriptors_list = ['MaxEStateIndex', 'MinEStateIndex', 'MaxPartialCharge',
                    'MinPartialCharge', 'FpDensityMorgan2', 'BalabanJ', 'BertzCT',
                    'HallKierAlpha', 'LabuteASA', 'TPSA', 'FractionCSP3', 'MolLogP']

def make_desc(smiles):
    mol = Chem.MolFromSmiles(smiles)
    calc = md.MolecularDescriptorCalculator(descriptors_list)
    funcs = calc.GetDescriptorFuncs()
    desc = [func(mol) for func in funcs]
    df = pd.DataFrame([desc], columns = descriptors_list, index = [smiles])
    return df

def categolize(df, category):
    df['category'] = category
    return df

def drop_category(df):
    for column in df.columns:
        if 'category' in column:
            df.drop(category, axis = 1)
    return df

def show_pca(df):
    data = df[descriptors_list]
    data = data.apply(lambda x: (x-x.mean())/x.std())
    pca = PCA(n_components = 2)
    df_pca = pd.DataFrame(pca.fit_transform(data), columns = ['PC1', 'PC2'], index = df.index)
    df_pca = pd.merge(df_pca, df, left_index = True, right_index = True)
    df_pca['category'] = df['category']

    fig = plt.figure()
    for i in df_pca['category'].unique():
        plt.scatter(df_pca[df_pca['category'] == i]['PC1'], df_pca[df_pca['category'] == i]['PC2'], label = i)
    plt.title('PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    return fig

def show_umap(df):
    data = df[descriptors_list]
    data = data.apply(lambda x: (x - x.mean()) / x.std())
    reducer = umap.UMAP(n_neighbors=150, random_state=41)
    reducer = reducer.fit(data)
    df_umap = pd.DataFrame(reducer.transform(data), columns=['axis 1', 'axis 2'], index=df.index)
    df_umap['category'] = df['category']

    fig = plt.figure()
    for i in df_umap['category'].unique():
        plt.scatter(df_umap[df_umap['category'] == i]['axis 1'], df_umap[df_umap['category'] == i]['axis 2'], label=i)
    plt.title('UMAP')
    plt.xlabel('axis 1')
    plt.ylabel('axis 2')
    plt.legend()
    return fig


def pred(smiless):
    df = []
    for smiles in smiless:
        mol = Chem.MolFromSmiles(smiles)
        ecfp = MakeECFP(mol)
        model = pickle.load(open('ML/model3.pickle', 'rb'))
        proba = model.predict_proba(ecfp)
        pred = model.predict(ecfp)
        re = [smiles, pred, proba[0][1], proba[0][0]]
        df.append(re)
    df = pd.DataFrame(df, columns = ['SMILES', 'prediction', 'probability (active)', 'probability (inactive)'])
    df['prediction'] = df['prediction'].map(lambda x: 'active' if x == 1 else 'inactive')
    return df