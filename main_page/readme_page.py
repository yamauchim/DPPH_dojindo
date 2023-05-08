import streamlit as st


def explain():
    st.header("取説")
    st.subheader("DPPH予測")
    st.markdown("SMILESを入力すると、Morgan Fingerprintに変換後、モデルに入力されます")
    st.subheader("Requirements")
    st.markdown("streamlit, rdkit-pypi, pandas, numpy, xgboost, scikit-learn, py3Dmol, stmol")
    st.text("A pickle file is necessary to use a new ML model.")
    st.text("If you want to make a ML model, please install matplotlib.")