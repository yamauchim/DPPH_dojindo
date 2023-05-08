import streamlit as st
import pandas as pd
from ML.func_y import make_desc, categolize, show_pca, show_umap

def visualize():
    st.header("visualize")
    st.markdown("FooDBで作ったUMAPにSMILESを入力した化合物を載せる")
    search_smiles = st.text_input("SMILESを入力", "Oc1ccccc1")
    path = "foodb_data_descriptors.csv"
    base_data = pd.read_csv(path)
    search_data = make_desc(search_smiles)
    base_data = categolize(base_data, 'base')
    search_data = categolize(search_data, 'search')
    data = pd.concat([base_data, search_data])
    st.dataframe(data)

    dec_meth = st.radio('方法の選択', ['UMAP(数分時間がかかります)', 'PCA'])

    if dec_meth == 'PCA':
        if st.button('Start'):
            fig_pca = show_pca(data)
            st.pyplot(fig_pca)

    if dec_meth == 'UMAP(数分時間がかかります)':
        if st.button('Start'):
            fig_umap = show_umap(data)
            st.pyplot(fig_umap)