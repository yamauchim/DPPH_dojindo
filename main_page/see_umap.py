import pandas as pd
import streamlit as st
from common.rdkitdescriptors import TransferSmiles, MakeDescriptors, make_desc_df
from ML import create_umap


def see_umap():
    st.header("UMAP")
    st.markdown("FooDBで作ったUMAPにSMILESを入力した化合物を載せる")
    search_smiles = st.text_input("SMILESを入力", "Oc1ccccc1")
    st.markdown("数分時間がかかります")
    path = "compound_mini.csv"
    base_data = pd.read_csv(path)

    if st.button("Start, good luck!"):
        # smilesを入力した化合物をUMAPプロットに載せて出
        desc = make_desc_df(base_data)
        st.dataframe(desc)
        added = create_umap.append(search_smiles, desc)
        cat = added["category"].to_list()
        added.drop("category", axis=1, inplace=True)
        std = create_umap.scale(added)
        st.dataframe(std)
        """
        #two_d = create_umap.pca_fit(std)
        reducer = create_umap.fit(std)
        two_d = create_umap.transform(std, reducer)
        two_d["category"] = cat
        st.pyplot(create_umap.show(two_d))"""