import shap
import pickle
from rdkit.Chem import AllChem
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

def view_ai(mol):
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, bitInfo=bi)
    model = pickle.load(open("ML/model3.pickle", "rb"))
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(fp)

    ai_list = np.zeros(mol.GetNumAtoms())
    for bit in bi:

        Cn = shap_values[0, bit]
        fn = len(bi[bit])

        for part in bi[bit]:
            if part[1] == 0:
                i = part[0]
                xn = 1
                ai_list[i] += Cn / fn / xn
            else:
                amap = {}
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius = part[1], rootedAtAtom = part[0])
                submol = Chem.PathToSubmol(mol, env, atomMap = amap)
                xn = len(amap.keys())

                for i in amap.keys():
                    ai_list[i] += Cn/ fn / xn
    ai_list = ai_list/abs(ai_list).max()

    for x in mol.GetAtoms():
        x.SetProp('atomNote', str(round(ai_list[x.GetIdx()], 3)))

    atoms = [i for i in range(len(ai_list))]
    atom_colors = dict()
    for i in atoms:
        if ai_list[i] > 0:
            atom_colors[i] = (1, 1 - ai_list[i], 1 - ai_list[i])
        else:
            atom_colors[i] = (1 + ai_list[i], 1 + ai_list[i], 1)

    view = rdMolDraw2D.MolDraw2DSVG(300, 350)
    tm = rdMolDraw2D.PrepareMolForDrawing(mol)
    view.DrawMolecule(tm,
                      highlightAtoms=atoms,
                      highlightAtomColors=atom_colors,
                      highlightBonds=[],
                      highlightBondColors={})

    view.FinishDrawing()
    svg = view.GetDrawingText()
    with open('highlighted_sample.svg', 'w') as f:
        f.write(svg)
    return svg
