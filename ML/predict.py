"""
def func(smiles, model):
    result = true or false
    return result
"""

from common.rdkitdescriptors import MakeECFP
import pickle
import xgboost #モデルインストール用


def Classifier(mol):
    # prepare a model with a pickle format
    ecfp = MakeECFP(mol)
    model = pickle.load(open("ML/model3.pickle", "rb")) # プルダウンで選ぶ
    pred = model.predict(ecfp)
    result = 0
    if pred == 1:
        result = 1
    return result

def showresultCLF(ans):
    if ans == 1:
        a_str = "Active"
    elif ans == 0:
        a_str = "Inactive"
    else:
        a_str = "Error"
    return a_str


