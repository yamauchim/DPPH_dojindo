from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.ML.Descriptors import MoleculeDescriptors
import pandas as pd
import numpy as np


def TransferSmiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol


def MakeECFP(mol):
    bi = {}
    fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, bitInfo=bi)
    ser_fp = pd.DataFrame(np.array(fp)).transpose()
    return ser_fp


def VisualizeSubmol(mol):
    bi = {}
    fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, bitInfo=bi)
    turples = ((mol, bit, bi) for bit in list(bi.keys()))
    drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
    pic = Draw.DrawMorganBits(turples, molsPerRow=4, legends=['bit: ' + str(x) for x in list(bi.keys())],
                              drawOptions=drawOptions)
    return pic


descriptors_list = ['MaxEStateIndex', 'MinEStateIndex', 'MaxPartialCharge',
                    'MinPartialCharge', 'FpDensityMorgan2', 'BalabanJ', 'BertzCT',
                    'HallKierAlpha', 'LabuteASA', 'TPSA', 'FractionCSP3', 'MolLogP']

def MakeDescriptors(data):
    # データを入力して記述子を作る
    base_df = data
    base_mol = [TransferSmiles(smi) for smi in base_df["SMILES"]]
    descriptor_calculation = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors_list)
    desc = [descriptor_calculation.CalcDescriptors(mol) for mol in base_mol]
    desc_df = pd.DataFrame(desc, columns=descriptors_list, index=base_df["SMILES"])
    return desc_df

def mol2feature(mol):
    descriptor_calculation = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors_list)
    desc = [descriptor_calculation.CalcDescriptors(mol)]
    desc_df = pd.DataFrame(desc, columns=descriptors_list)
    return desc_df
# 単数データをumapに入れるのは無理


def make_desc_df(df):
    mols = []
    for smiles in df['SMILES']:
        try:
            mols.append(Chem.MolFromSmiles(smiles))
        except:
            mols.append('error')
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors_list)
    funcs = calculator.GetDescriptorFuncs()
    descs = []
    for mol in mols:
        desc = []
        for func in funcs:
            try:
                desc.append(func(mol))
            except:
                desc.append(0)
        descs.append(desc)
    desc_df = pd.DataFrame(descs, columns = descriptors_list, index = df['SMILES'])
    return df