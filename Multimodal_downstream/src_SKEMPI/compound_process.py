import numpy as np
import os
import pickle
from rdkit import Chem
from rdkit.Chem import MolFromSmiles

degrees = [0, 1, 2, 3, 4, 5]

class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]

class MolGraph(object):
    def __init__(self):
        self.nodes = {} # dict of lists of nodes, keyed by node type

    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)

        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=True):
    if bool_id_feat:
        pass
        # return np.array([atom_to_id(atom)])
    else:
        results = one_of_k_encoding_unk(
          atom.GetSymbol(),
          [
            'B',
            'C',
            'N',
            'O',
            'F',
            'Si',
            'P',
            'S',
            'Cl',
            'As',
            'Se',
            'Br',
            'Te',
            'I',
            'At',
            'other'
          ]) + one_of_k_encoding(atom.GetDegree(),
                                 [0, 1, 2, 3, 4, 5]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2,'other'
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]

        return np.array(results)

def bond_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats)

def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        # atom_features(atom) : array([0,1,0,0,...]), size:39;   atom.GetIdx() : int, take value from 0 to n_atom
        # new_atom_node : class(Node) : def __init__(self, ntype, features, rdkit_ix) : self.ntype = 'atom', self.features = features, self._neighbors = [], self.rdkit_ix = rdkit_ix
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom.GetIdx())
        # atoms_by_rd_idx : {0: class(Node), 1: class(Node), 2: class(Node), ..., }
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node

    for bond in mol.GetBonds():
        # Finding begin_atom and end_atom of each bond
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]  # class(Node)
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]  # class(Node)
        # bond_features(bond) : array([True, False, ..., ])
        # new_bond_node : class(Node) : def __init__(self, ntype, features, rdkit_ix) : self.ntype = 'bond', self.features = features, self._neighbors = [], self.rdkit_ix = None
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        # new_bond_node : class(Node) : ... : self.ntype = 'bond', self.features = features, self._neighbors = [atom1_node, atom2_node], self.rdkit_ix = None.
        #                                     and atom1_node._neighbors.append(new_bond_node), atom2_node._neighbors.append(new_bond_node)
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        # atom1_node : class(Node) : ... : self.ntype = 'atom', self.features = features, self._neighbors = [atom2_node], self.rdkit_ix = rdkit_ix.
        #                                  and atom2_node._neighbors.append(atom1_node)
        atom1_node.add_neighbors((atom2_node,))
    # mol_node : class(Node) : ... : self.ntype = 'molecule', self.features = None, self._neighbors = [], self.rdkit_ix = None.
    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph

def array_rep_from_smiles(molgraph):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    # molgraph = graph_from_smiles_tuple(tuple(smiles))
    degrees = [0,1,2,3,4,5]
    # 'atom_features' : array([[0,1,0,0,...], [0,0,0,1,...], ...])
    # 'bond_features' : array([[True,False,False,False,...], [True,False,False,True,...], ...])
    # 'atom_list' : list([[0,13,26,14,...]]), all atoms adjacent to a molecule
    # 'rdkit_ix' : array([0,9,13,15,...]), rdkit_ix of all atoms
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'),
                'rdkit_ix'      : molgraph.rdkit_ix_array()}

    for degree in degrees:
        # ('atom_neighbors', degree) : array([[], [], [], ...]), neighbors of which type is atom for each atom
        # ('bond_neighbors', degree) : array([[], [], [], ...]), neighbors of which type is bond for each atom
        arrayrep[('atom_neighbors', degree)] = np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep


def gen_descriptor_data(smilesList):
    smiles_to_fingerprint_array = {}
    for i, smiles in enumerate(smilesList):
#         if i > 5:
#             print("Due to the limited computational resource, submission with more than 5 molecules will not be processed")
#             break
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        try:
            # molgraph : class(MolGraph), includeing atom nodes, molecule nodes, bond nodes and their neighbouring relations
            molgraph = graph_from_smiles(smiles)
            # molgraph['atom'] is reordered from small to large
            molgraph.sort_nodes_by_degree('atom')
            # arrayrep : dict:{
            #    'atom_features' : array([[0,1,0,0,...], [0,0,0,1,...], ...])
            #    'bond_features' : array([[True,False,False,False,...], [True,False,False,True,...], ...])
            #    'atom_list' : list([[0,13,26,14,...]]), all atoms adjacent to a molecule
            #    'rdkit_ix' : array([0,9,13,15,...]), rdkit_ix of all atoms
            #    ('atom_neighbors', degree) : array([[], [], [], ...]), neighbors of which type is atom for each atom
            #    ('bond_neighbors', degree) : array([[], [], [], ...]), neighbors of which type is bond for each atom
            # }
            arrayrep = array_rep_from_smiles(molgraph)
            smiles_to_fingerprint_array[smiles] = arrayrep
            
        except:
            print(smiles)
    return smiles_to_fingerprint_array

def get_smiles_array(smilesList, feature_dicts):
    x_mask = []
    x_atom = []
    x_bonds = []
    x_atom_index = []
    x_bond_index = []
    for smiles in smilesList:
        x_mask.append(feature_dicts['smiles_to_atom_mask'][smiles])
        x_atom.append(feature_dicts['smiles_to_atom_info'][smiles])
        x_bonds.append(feature_dicts['smiles_to_bond_info'][smiles])
        x_atom_index.append(feature_dicts['smiles_to_atom_neighbors'][smiles])
        x_bond_index.append(feature_dicts['smiles_to_bond_neighbors'][smiles])
    return np.asarray(x_atom),np.asarray(x_bonds),np.asarray(x_atom_index),\
        np.asarray(x_bond_index),np.asarray(x_mask),feature_dicts['smiles_to_rdkit_list']

def save_smiles_dicts(smilesList):
    """
    smilesList : array(['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...]), 13464
    """
    # first need to get the max atom length
    max_atom_len = 0
    max_bond_len = 0
    num_atom_features = 0
    num_bond_features = 0
    smiles_to_rdkit_list = {}
    # smiles_to_fingerprint_features : dict({smiles: arrayrep, smiles: arrayrep, ...})
    # arrayrep : dict:{
    #    'atom_features' : array([[0,1,0,0,...], [0,0,0,1,...], ...]), array(atom_num, atom_features_size) eg.array(39,39)
    #    'bond_features' : array([[True,False,False,False,...], [True,False,False,True,...], ...]), array(bond_num, bond_features_size) eg.array(39,10)
    #    'atom_list' : list([[0,13,26,14,...]]), all atoms adjacent to a molecule, list([atom_num]) eg.list([39])
    #    'rdkit_ix' : array([0,9,13,15,...]), rdkit_ix of all atoms, array(atom_num) eg.array(39)
    #    ('atom_neighbors', degree) : array([[], [], [], ...]), neighbors of which type is atom for each atom
    #    ('bond_neighbors', degree) : array([[], [], [], ...]), neighbors of which type is bond for each atom
    # }
    smiles_to_fingerprint_features = gen_descriptor_data(smilesList)
    # smiles_to_rdkit_list : dict({smiles: array([0,11,14,...]), smiles: array([0,5,6,8,...]), ...})
    for smiles,arrayrep in smiles_to_fingerprint_features.items():
        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        rdkit_list = arrayrep['rdkit_ix']
        smiles_to_rdkit_list[smiles] = rdkit_list 

        atom_len,num_atom_features = atom_features.shape
        bond_len,num_bond_features = bond_features.shape

        if atom_len > max_atom_len:
            max_atom_len = atom_len
        if bond_len > max_bond_len:
            max_bond_len = bond_len

    # then add 1 so I can zero pad everything
    max_atom_index_num = max_atom_len
    max_bond_index_num = max_bond_len

    max_atom_len += 1
    max_bond_len += 1

    smiles_to_atom_info = {}
    smiles_to_bond_info = {}

    smiles_to_atom_neighbors = {}
    smiles_to_bond_neighbors = {}

    smiles_to_atom_mask = {}

    degrees = [0,1,2,3,4,5]
    # then run through our numpy array again
    for smiles,arrayrep in smiles_to_fingerprint_features.items():
        mask = np.zeros((max_atom_len))

        # get the basic info of what
        # my atoms and bonds are initialized
        atoms = np.zeros((max_atom_len,num_atom_features))
        bonds = np.zeros((max_bond_len,num_bond_features))

        # then get the arrays initlialized for the neighbors
        atom_neighbors = np.zeros((max_atom_len,len(degrees)))
        bond_neighbors = np.zeros((max_atom_len,len(degrees)))

        # now set these all to the last element of the list, which is zero padded
        atom_neighbors.fill(max_atom_index_num)
        bond_neighbors.fill(max_bond_index_num)

        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']
        
        for i,feature in enumerate(atom_features):
            mask[i] = 1.0
            atoms[i] = feature

        for j,feature in enumerate(bond_features):
            bonds[j] = feature
        
        atom_neighbor_count = 0
        bond_neighbor_count = 0
        working_atom_list = []
        working_bond_list = []
        for degree in degrees:
            atom_neighbors_list = arrayrep[('atom_neighbors', degree)]
            bond_neighbors_list = arrayrep[('bond_neighbors', degree)]

            if len(atom_neighbors_list) > 0:

                for i,degree_array in enumerate(atom_neighbors_list):
                    for j,value in enumerate(degree_array):
                        atom_neighbors[atom_neighbor_count,j] = value
                    atom_neighbor_count += 1

            if len(bond_neighbors_list) > 0:
                for i,degree_array in enumerate(bond_neighbors_list):
                    for j,value in enumerate(degree_array):
                        bond_neighbors[bond_neighbor_count,j] = value
                    bond_neighbor_count += 1
        
        # then add everything to my arrays   
        # atoms : array([[0,1,0,...,0],[0,0,0,1,...,0],...,[0,0,0,..,0]]) size:(max_atom_len,num_atom_features) eg.(178,39)
        # bonds : array([[1,0,0,...,0],[1,0,0,1,...,0],...,[0,0,0,..,0]]) size:(max_bond_len,num_atom_features) eg.(182,10)
        smiles_to_atom_info[smiles] = atoms
        smiles_to_bond_info[smiles] = bonds
        # atom_neighbors : array([[13,177,177,...],[28,177,177,...],...,[0,26,177,177,...],[14,16,177,177,...],...]) eg.(178,6)
        # bond_neighbors : array([[0,181,181,...],[8,181,181,...],...,[0,1,181,181,...],[2,3,181,181,...],...]) eg.(178,6)
        smiles_to_atom_neighbors[smiles] = atom_neighbors
        smiles_to_bond_neighbors[smiles] = bond_neighbors
        # mask : array([1,1,1,...,1,0,0,0,....]) len : max_atom_len
        smiles_to_atom_mask[smiles] = mask

    del smiles_to_fingerprint_features
    feature_dicts = {
        'smiles_to_atom_mask': smiles_to_atom_mask,
        'smiles_to_atom_info': smiles_to_atom_info,
        'smiles_to_bond_info': smiles_to_bond_info,
        'smiles_to_atom_neighbors': smiles_to_atom_neighbors,
        'smiles_to_bond_neighbors': smiles_to_bond_neighbors,
        'smiles_to_rdkit_list': smiles_to_rdkit_list
    }
    return feature_dicts

def search_problem_smiles(smilesList, smiles_feature_dict_path):
    # feature_dicts(dict) : 
    #     {
    #       'smiles_to_atom_mask': smiles_to_atom_mask,
    #       'smiles_to_atom_info': smiles_to_atom_info,
    #       'smiles_to_bond_info': smiles_to_bond_info,
    #       'smiles_to_atom_neighbors': smiles_to_atom_neighbors,
    #       'smiles_to_bond_neighbors': smiles_to_bond_neighbors,
    #       'smiles_to_rdkit_list': smiles_to_rdkit_list
    #     }
    # smiles_to_atom_mask : 
    #     {
    #        smiles: array([1,1,1,...,1,0,0,0,....]) len: max_atom_len, 0 is padding
    #        smiles: ...,
    #        ...,
    #     }
    # smiles_to_atom_info : 
    #     {
    #        smiles: array([[0,1,0,...,0],[0,0,0,1,...,0],...,[0,0,0,..,0]]) size:(max_atom_len,num_atom_features) eg.(178,39), 
    #        smiles: ...,
    #        ...,
    #     }
    # smiles_to_bond_info : 
    #     {
    #        smiles: array([[1,0,0,...,0],[1,0,0,1,...,0],...,[0,0,0,..,0]]) size:(max_bond_len,num_atom_features) eg.(182,10), 
    #        smiles: ...,
    #        ...,
    #     }
    # smiles_to_atom_neighbors : 
    #     {
    #        smiles: array([[13,177,177,...],[28,177,177,...],...,[0,26,177,177,...],[14,16,177,177,...],...]) eg.(178,6), 177 is padding
    #        smiles: ...,
    #        ...,
    #     }
    # smiles_to_bond_neighbors : 
    #     {
    #        smiles: array([[0,181,181,...],[8,181,181,...],...,[0,1,181,181,...],[2,3,181,181,...],...]) eg.(178,6), 181 is padding
    #        smiles: ...,
    #        ...,
    #     }
    # smiles_to_rdkit_list : 
    #     {
    #        smiles: array([0,9,13,15,...]), rdkit_ix of all atoms, array(atom_len) eg.array(39)
    #        smiles: ...,
    #        ...,
    #     }
    if os.path.exists(smiles_feature_dict_path):
        smiles_feature_dict = pickle.load(open(smiles_feature_dict_path, "rb"))
    else:
        atom_num_dist = []
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                mol = Chem.MolFromSmiles(smiles)  # input : smiles seqs, output : molecule obeject
                atom_num_dist.append(len(mol.GetAtoms()))  # list : get atoms obeject num from molecule obeject
                remained_smiles.append(smiles)  # list : smiles without transformation error
                canonical_smiles_list.append(Chem.MolToSmiles(mol, isomericSmiles=True))  # canonical smiles without transformation error
            except:
                print("the smile \"%s\" has transformation error in the first test" % smiles)
                pass
        print("number of successfully processed smiles after test: %d/%d" % (len(remained_smiles), len(smilesList)))

        # smilesList : ['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...], 13464
        smiles_feature_dict = save_smiles_dicts(remained_smiles)
        with open(smiles_feature_dict_path, 'wb') as f:
            pickle.dump(smiles_feature_dict, f)

    problem_indexs = []
    for index, smiles in enumerate(smilesList):
        if smiles not in smiles_feature_dict['smiles_to_atom_mask']:
            problem_indexs.append(index)

    return smiles_feature_dict, problem_indexs
        
def load_degree_smiles(smilesList, smiles_feature_dict_path):
    if os.path.exists(smiles_feature_dict_path):
        smiles_feature_dict = pickle.load(open(smiles_feature_dict_path, "rb"))
    else:
        atom_num_dist = []
        remained_smiles = []
        canonical_smiles_list = []
        for smiles in smilesList:
            try:
                mol = Chem.MolFromSmiles(smiles)  # input : smiles seqs, output : molecule obeject
                atom_num_dist.append(len(mol.GetAtoms()))  # list : get atoms obeject num from molecule obeject
                remained_smiles.append(smiles)  # list : smiles without transformation error
                canonical_smiles_list.append(Chem.MolToSmiles(mol, isomericSmiles=True))  # canonical smiles without transformation error
            except:
                print("the smile \"%s\" has transformation error in the first test" % smiles)
                pass
        print("number of successfully processed smiles after test: %d/%d" % (len(remained_smiles), len(smilesList)))

        # smilesList : ['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...], 13464
        smiles_feature_dict = save_smiles_dicts(remained_smiles)
        with open(smiles_feature_dict_path, 'wb') as f:
            pickle.dump(smiles_feature_dict, f)
            
    # x_atom：array, (batch_size, max_atom_len, num_atom_features)
    # x_bond: array, (batch_size, max_bond_len, num_bond_features)
    # x_atom_degree: array, (batch_size, max_atom_len, degrees_len)
    # x_bond_degree: array, (batch_size, max_bond_len, degrees_len)
    # x_mask: array, array, (batch_size, max_atom_len)
    # smiles_to_rdkit_list: dict, {smiles: array([atom_len]), smiles: array([atom_len]), ...,}  len: all_atoms_num
    x_atom, x_bond, x_atom_degree, x_bond_degree, x_mask, smiles_to_rdkit_list = get_smiles_array(smilesList, smiles_feature_dict)

    return x_atom, x_bond, x_atom_degree, x_bond_degree, x_mask

