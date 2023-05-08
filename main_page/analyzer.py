import streamlit as st
from common.rdkitdescriptors import TransferSmiles, VisualizeSubmol
from common import stmolblock
from ML import predict, func_y
from rdkit import Chem

from ML import shap_test

def predict_dpph():
    st.markdown("# DPPH analyzer")
    st.markdown("Produced by --")

    search_smiles = st.text_input("SMILESを入力", "CC(C)(C)C1=CC(=C(C(=C1)C(C)(C)C)O)C(C)(C)C")  # 文言とデフォルト値
    input_mol = Chem.MolFromSmiles(search_smiles)
    if st.button("Start"):
        pred = func_y.pred([search_smiles])
        st.dataframe(pred)
        st.text('原子毎の予測への寄与')
        st.image(shap_test.view_ai(input_mol))
    st.markdown("入力構造")
    blk = stmolblock.makeblock(input_mol)
    stmolblock.render_mol(blk)
    if st.button('入力ビットの表示'):
        st.image(VisualizeSubmol(input_mol))