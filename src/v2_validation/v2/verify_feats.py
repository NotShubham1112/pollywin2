import json, ast, numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, GraphDescriptors, rdMolDescriptors

nb = json.load(open(r'D:\Parth\ploywin r2\PolyWin_R2_v14_p1m_pretrain.ipynb', encoding='utf-8'))
src = ''.join(nb['cells'][4]['source'])
tree = ast.parse(src)

feats_fn = None
fnames = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == 'feats':
        feats_fn = ast.get_source_segment(src, node)
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == 'FNAMES':
                fnames = ast.get_source_segment(src, node.value)

# Build namespace for exec
ns = {'np': np, 'Chem': Chem, 'Descriptors': Descriptors, 'Crippen': Crippen,
      'GraphDescriptors': GraphDescriptors, 'rdMolDescriptors': rdMolDescriptors}
# inject names referenced in feats body that come from aliases in the notebook
# The feats body uses: Descriptors.MolWt, rdMolDescriptors.CalcNumLipinskiHBA etc.
exec(feats_fn, ns)
exec('FNAMES = ' + fnames, ns)

# Test on several real SMILES
smiles_list = ['CCO', 'c1ccccc1C(=O)O', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
               'O=C(O)C1CCCCC1C(=O)O', 'Cc1ccccc1C(C)C',
               'C1=CC(=C(C=C1Cl)Cl)Cl', 'COc1ccc2cc(ccc2c1)OC']
for s in smiles_list:
    m = Chem.MolFromSmiles(s)
    f = ns['feats'](m)
    assert len(f) == len(ns['FNAMES']), (s, len(f), len(ns['FNAMES']))
print('feats() count == FNAMES count ==', len(ns['FNAMES']))

# verify None handling
f0 = ns['feats'](None)
print('None mol padding count:', len(f0))
print('FNAMES len:', len(ns['FNAMES']))
print('ALL OK: len(feats(m)) == len(FNAMES) ==', len(f0))