import streamlit as st
import pandas as pd
from ML import predict, func_y
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import PandasTools


def predict_dpph_multi():
    st.markdown("# DPPH analyzer")

    smiles_file = st.file_uploader('smilesのcsvファイルをアップロードしてください。', type = 'csv')
    with open('smiles_example.csv', 'rb')as file:
        st.download_button('Download a sample file', file, 'sample.csv')

    if st.button('Start'):
        df = pd.read_csv(smiles_file)
        errors = []
        keys = []
        if 'SMILES' in df:
            smiless = df['SMILES']
        elif 'smiles' in df:
            smiless = df['smiles']
            df = df.rename(columns = {'smiles': 'SMILES'})
        else:
            smiless = []
            for column in df:
                for n, smiles in enumerate(df[column]):
                    try:
                        smiless.append(smiles)
                        keys.append(n)
                    except:
                        errors.append(smiles)


        re = func_y.pred(smiless)
        df = pd.merge(df, re, on='SMILES', how='left')
        df_sort = df.sort_values('probability (active)', ascending=False)
        st.dataframe(df_sort)
        csv = df_sort.to_csv().encode('utf-8')
        st.download_button('Download', csv, 'predictions.csv')

        """else:
            st.text('error')"""
