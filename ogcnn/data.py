import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
import os, sys
import math
import re
import functools
from pymatgen.core.structure import Structure,SiteCollection
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.analysis.local_env import VoronoiNN,MinimumDistanceNN
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read,write
from ase.neighborlist import natural_cutoffs
from ase.data import atomic_numbers, covalent_radii, atomic_masses
from ase.neighborlist import NewPrimitiveNeighborList
from torch.utils.data import Sampler
import math

#Original batch sampler for even distribution of adsorption energy data
class BalancedBatchSampler(Sampler):
    def __init__(self, attribute_indices, batch_size):
        self.attribute_indices = attribute_indices
        self.batch_size = batch_size
        self.attributes = list(attribute_indices.keys())
        self.ratio = [len(attribute_indices[key]) for key in self.attributes]
        self.s = sum(self.ratio)
        self.ratio = [i/self.s for i in self.ratio]
        self.num_attributes = len(self.attributes)
        self.samples_per_attr={}
        n=0
        for i,attr in enumerate(self.attributes[:-1]):
            self.samples_per_attr[attr] = int(batch_size*self.ratio[i])
            n += int(batch_size*self.ratio[i])
        self.samples_per_attr[self.attributes[-1]]=batch_size-n
        # self.attribute_iterators = {attr: iter(indices) for attr, indices in self.attribute_indices.items()}
        self.attribute_iterators = {attr: iter(np.random.permutation(indices)) for attr, indices in self.attribute_indices.items()}

    def __iter__(self):
        while True:
            batch = []
            for attr in self.attributes:
                try:
                    for _ in range(self.samples_per_attr[attr]):
                        idx = next(self.attribute_iterators[attr])
                        batch.append(idx)
                except:
                    self.attribute_iterators = {attr: iter(np.random.permutation(indices)) for attr, indices in self.attribute_indices.items()}
                    StopIteration
                    # その属性のデータが尽きた場合は終了
                    return
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        # return int(self.s/self.batch_size)
        return math.ceil(self.s / self.batch_size)
    

def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, attribute_indices=None,**kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print('[Warning] train_ratio is None, using all training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1
    indices = list(range(total_size))
    if kwargs['train_size']:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if kwargs['test_size']:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if kwargs['val_size']:
        valid_size = kwargs['val_size']
    else:
        valid_size = int(val_ratio * total_size)
    
    train_sampler = BalancedBatchSampler(attribute_indices={key: item[:int(item.size*train_ratio)] for key,item in attribute_indices.items()},batch_size=batch_size)
    train_loader = DataLoader(dataset, 
                            #sampler=train_sampler,
                            batch_sampler=train_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    
    if val_ratio==0:
        val_loader = DataLoader(dataset, 
                            sampler=SubsetRandomSampler(indices[:0]),
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
        test_loader = DataLoader(dataset, 
                            sampler=SubsetRandomSampler(indices[:0]),
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
        if return_test:
            return train_loader, val_loader, test_loader
        else:
            return train_loader, val_loader

    val_sampler = BalancedBatchSampler(attribute_indices={key: item[-int(item.size*val_ratio):] for key,item in attribute_indices.items()},batch_size=batch_size)
    val_loader = DataLoader(dataset, 
                            #sampler=val_sampler,
                            batch_sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        if test_ratio==0:
            test_loader = DataLoader(dataset, 
                            sampler=SubsetRandomSampler(indices[:0]),
                            batch_size=batch_size,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
            return train_loader, val_loader, test_loader
        else:        
            test_sampler = BalancedBatchSampler(attribute_indices={key: item[int(item.size*train_ratio):-int(item.size*val_ratio)] for key,item in attribute_indices.items()},batch_size=batch_size)
            test_loader = DataLoader(dataset, 
                                    #sampler=test_sampler,
                                    batch_sampler=test_sampler,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn, pin_memory=pin_memory)
            return train_loader, val_loader, test_loader
    
    else:
        return train_loader, val_loader



def collate_pool(dataset_list): # batch the crystal atoms
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
        ([atom_fea, hot_fea], nbr_fea, nbr_fea_idx, target)

        atom_fea: torch.Tensor shape(n_i, atom_fea_len)
        hot_fea: torch.Tensor shape(n_i, hot_fea_len)
        nbr_fea: torch.Tensor shape(n_i, nbr_fea_len)
        nbr_fea_idx: torch.LongTensor shape(n_i, M)
        target: torch.Tensor shape(1,)
        cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_hot_fea: torch.Tensor shape (N, hot_fea_len)
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_hot_fea = [], [], [], []
    crystal_atom_idx, crystal_atom_idx1, batch_target, batch_target1, batch_target2 = [], [], [], [], []
    batch_cif_ids = []
    base_idx = 0
    for i, (([atom_fea,hot_fea],nbr_fea,nbr_fea_idx,ads_idx, site_idx),target, target1, target2, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_hot_fea.append(hot_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)
        # new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        new_idx = ads_idx + base_idx
        new_idx1 = site_idx + base_idx
        crystal_atom_idx.append(new_idx)
        crystal_atom_idx1.append(new_idx1)
        batch_target.append(target)
        batch_target1.append(target1)
        batch_target2.append(target2)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return ([torch.cat(batch_atom_fea, dim=0),
             torch.cat(batch_hot_fea, dim=0)],
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
            crystal_atom_idx1),\
        torch.stack(batch_target, dim=0),\
        torch.cat(batch_target1, dim=0),\
        torch.stack(batch_target2, dim=0),\
        batch_cif_ids

def calculateDistance(a,b):   # Atom-wise OFM
    dist =math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    return dist

def make_hot_for_atom_i(atoms,i,HV_K,nbs):
    HV_P = HV_K[i].T
    AA = HV_P.reshape((HV_P.shape[1], 32))
    A = np.array(AA)

    HV_K = np.sum(HV_K[nbs],axis=0)
    X0 = np.matmul(HV_P, HV_K)
    X0  = np.concatenate((A.T,X0),axis = 1)
    X0 = np.asarray(X0)
    return X0 

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------
        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        ##print("Filter:",self.filter,len(self.filter))
        ##print("Var:",self.var)
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)
    
class ConnectivityType(object):
    """
    Connectivity with type

    Unit: angstrom
    """
    def __init__(self, vdw_min):
        
        # assert dmin < dmax
        #assert dmax - dmin > step
        self.vdw_min = vdw_min

    def expand(self, distances, atoms, nbr_fea_idx):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        tmp = np.shape(distances)
        tmp_dist = np.zeros([tmp[0],tmp[1],4])
        numbers = atoms.get_atomic_numbers()
        for i in range(tmp[0]):
            e1 = numbers[i]
            for j in range(tmp[1]):
                idx=nbr_fea_idx[i,j]
                e2 = numbers[idx]
                # 4 categories, 0: organic-organic, 1: organic-metal, 2: metal-metal, 3: long range
                if np.isnan(distances[i,j]):
                    continue
                if e1 > 10 and e2 > 10:
                    tmp_dist[i,j,2] = 1.0
                elif e1 <=10 and e2 <= 10:
                    if distances[i,j] < self.vdw_min:
                        tmp_dist[i,j,0] = 1.0
                    else:
                        tmp_dist[i,j,3] = 1.0
                else:
                    tmp_dist[i,j,1] = 1.0
                    
                       
                    ##crystal[i].specie.number
        return tmp_dist

class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        ##print(atom_types)
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

elements = {'H':['1s2'],'Li':['[He] 1s2'],'Be':['[He] 2s2'],'B':['[He] 2s2 2p1'],'N':['[He] 2s2 2p3'],'O':['[He] 2s2 2p4'],
                     'C':['[He] 2s2 2p2'], 'I':['[Kr] 4d10 5s2 5p5'],
                     'F':['[He] 2s2 2p5'],'Na':['[Ne] 3s1'],'Mg':['[Ne] 3s2'],'Al':['[Ne] 3s2 3p1'],'Si':['[Ne] 3s2 3p2'],
                     'P':['[Ne] 3s2 3p3'],'S':['[Ne] 3s2 3p4'],'Cl':['[Ne] 3s2 3p5'],'K':['[Ar] 4s1'],'Ca':['[Ar] 4s2'],'Sc':['[Ar] 3d1 4s2'],
                     'Ti':['[Ar] 3d2 4s2'],'V':['[Ar] 3d3 4s2'],'Cr':['[Ar] 3d5 4s1'],'Mn':['[Ar] 3d5 4s2'],
                     'Fe':['[Ar] 3d6 4s2'],'Co':['[Ar] 3d7 4s2'],'Ni':['[Ar] 3d8 4s2'],'Cu':['[Ar] 3d10 4s1'],'Zn':['[Ar] 3d10 4s2'],
                     'Ga':['[Ar] 3d10 4s2 4p2'],'Ge':['[Ar] 3d10 4s2 4p2'],'As':['[Ar] 3d10 4s2 4p3'],'Se':['[Ar] 3d10 4s2 4p4'],'Br':['[Ar] 3d10 4s2 4p5'],'Rb':['[Kr] 5s1'],
                     'Sr':['[Kr] 5s2'],'Y':['[Kr] 4d1 5s2'],'Zr':['[Kr] 4d2 5s2'],'Nb':['[Kr] 4d4 5s1'],'Mo':['[Kr] 4d5 5s1'],
                     'Ru':['[Kr] 4d7 5s1'],'Rh':['[Kr] 4d8 5s1'],'Pd':['[Kr] 4d10'],'Ag':['[Kr] 4d10 5s1'],'Cd':['[Kr] 4d10 5s2'],
                     'In':['[Kr] 4d10 5s2 5p1'],'Sn':['[Kr] 4d10 5s2 5p2'],'Sb':['[Kr] 4d10 5s2 5p3'],'Te':['[Kr] 4d10 5s2 5p4'],'Cs':['[Xe] 6s1'],'Ba':['[Xe] 6s2'],
                     'La':['[Xe] 5d1 6s2'],'Ce':['[Xe] 4f1 5d1 6s2'],'Hf':['[Xe] 4f14 5d2 6s2'],'Ta':['[Xe] 4f14 5d3 6s2'],
                     'W':['[Xe] 4f14 5d5 6s1'],'Re':['[Xe] 4f14 5d5 6s2'],'Os':['[Xe] 4f14 5d6 6s2'],
                     'Ir':['[Xe] 4f14 5d7 6s2'],'Pt':['[Xe] 4f14 5d10'],'Au':['[Xe] 4f14 5d10 6s1'],'Hg':['[Xe] 4f14 5d10 6s2'],
                     'Tl':['[Xe] 4f14 5d10 6s2 6p2'],'Pb':['[Xe] 4f14 5d10 6s2 6p2'],'Bi':['[Xe] 4f14 5d10 6s2 6p3'],
                     'Tc':['[Kr] 4d5 5s2'],'Fr':['[Rn]7s1'],'Ra':['[Rn]7s2'],'Pr':['[Xe]4f3 6s2'],
                     'Nd':['[Xe] 4f4 6s2'],'Pm':['[Xe] 4f5 6s2'],'Sm':['[Xe] 4f6 6s2'],
                     'Eu':['[Xe] 4f7 6s2'],'Gd':['[Xe] 4f7 5d1 6s2'],'Tb':['[Xe] 4f9 6s2'],
                     'Dy':['[Xe] 4f10 6s2'],'Ho':['[Xe] 4f11 6s2'],'Er':['[Xe] 4f12 6s2'],
                     'Tm':['[Xe] 4f13 6s2'],'Yb':['[Xe] 4f14 6s2'],'Lu':['[Xe] 4f14 5d1 6s2'],
                     'Po':['[Xe] 4f14 5d10 6s2 6p4'],'At':['[Xe] 4f14 5d10 6s2 6p5'],
                     'Ac':['[Rn] 6d1 7s2'],'Th':['[Rn] 6d2 7s2'],'Pa':['[Rn] 5f2 6d1 7s2'],
                     'U':['[Rn] 5f3 6d1 7s2'],'Np':['[Rn] 5f4 6d1 7s2'],'Pu':['[Rn] 5f6 7s2'],
                     'Am':['[Rn] 5f7 7s2'],'Cm':['[Rn] 5f7 6d1 7s2'],'Bk':['[Rn] 5f9 7s2'],
                     'Cf':['[Rn] 5f10 7s2'],'Es':['[Rn] 5f11 7s2'],'Fm':['[Rn] 5f12 7s2'],
                     'Md':['[Rn] 5f13 7s2'],'No':['[Rn] 5f14 7s2'],'Lr':['[Rn] 5f14 6d1 7s2'],
                     'Rf':['[Rn] 5f14 6d2 7s2'],'Db':['[Rn] 5f14 6d3 7s2'],
                     'Sg':['[Rn] 5f14 6d4 7s2'],'Bh':['[Rn] 5f14 6d5 7s2'],
                     'Hs':['[Rn] 5f14 6d6 7s2'],'Mt':['[Rn] 5f14 6d7 7s2'],'Xe': ['[Kr] 4d10 5s2 5p6'], 'He':['1s2'], 'Kr':['[Ar] 3d10 4s2 4p6'], 'Ar': ['[Ne] 3s2 3p6'], 'Ne':['[He] 2s2 2p6']}
orbitals = {"s1":0,"s2":1,"p1":2,"p2":3,"p3":4,"p4":5,"p5":6,"p6":7,"d1":8,"d2":9,"d3":10,"d4":11,
    "d5":12,"d6":13,"d7":14,"d8":15,"d9":16,"d10":17,"f1":18,"f2":19,"f3":20,"f4":21,
    "f5":22,"f6":23,"f7":24,"f8":25,"f9":26,"f10":27,"f11":28,"f12":29,"f13":30,"f14":31}
hv = np.zeros(shape=(32,1))
hvs = {}
for key in elements.keys():
    element = key
    hv = np.zeros(shape=(32,1))
    s = elements[key][0]
    sp = (re.split('(\s+)', s))
    if key == "H":
        hv[0] = 1
    if key != "H":
        for j in range(1,len(sp)):
            if sp[j] != ' ':
                n = sp[j][:1]
                orb = sp[j][1:]
                hv[orbitals[orb]] = 1
    hvs[element] = hv

class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    or molecules are stored in the form of an .xyz trajectory or ASE Atoms list.
    (Note: Despite the name, it processes .xyz structures/models for BE-OGCNN).

    Parameters
    ----------

    root_dir: str or list of ase.Atoms
        The path to the .xyz file or an already parsed list of ase.Atoms objects.
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing Gaussian distances
    step: float
        The step size for constructing Gaussian distances
    orbital: bool
        Whether to generate and append orbital specific hidden features

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=4, dmin=0, step=0.2, orbital=False,
                 random_seed=2):
        # self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        base_dir = os.path.dirname(os.path.abspath(__file__))
        atom_init_file = os.path.join(base_dir, 'atom_init.json')
        if type(root_dir) == str:
            self.atoms_list = read(os.path.join(os.getcwd(), 'data_demo',root_dir),index=':')
        else: self.atoms_list = root_dir
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.ctf = ConnectivityType(vdw_min=1.25) 
        self.orbital = orbital

    def __len__(self):
        return len(self.atoms_list)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id = idx
        atoms = self.atoms_list[idx]
        atom_elem_nums = atoms.get_atomic_numbers()
        ads_idx = np.where(atom_elem_nums<10)[0]
        if ads_idx.size == 0:
            ads_idx = np.array([0])
        ads_idx = torch.LongTensor(ads_idx)
        target = atoms.info.get('ad',np.nan)
        env = atoms.info.get('env', 'in-house')
        if 'd_band_centers' in atoms.arrays.keys():
            target = atoms.arrays['d_band_centers']
        else:
            target = np.array([np.nan]*len(atoms))
        if atoms.calc is None:
            target2 = np.nan
        else:
            target2 = atoms.get_total_energy()/len(atoms)
        
        atom_fea = np.vstack([self.ari.get_atom_fea(atoms.get_atomic_numbers()[i])
                              for i in range(len(atoms))])
   
        atom_fea = torch.Tensor(atom_fea)
        neighbor_list = NewPrimitiveNeighborList(cutoffs=natural_cutoffs(atoms),self_interaction=False, bothways=True,sorted=True)
        neighbor_list.update(atoms.pbc, atoms.get_cell(), atoms.positions)

        if self.orbital:
            coodinates = np.array([len(np.unique(neighbor_list.get_neighbors(i)[0])) for i in range(len(atoms))])
            HV_K = [hvs[EK] for EK in atoms.get_chemical_symbols()]
            HV_K = [tmp.reshape((tmp.shape[1], 32)) for tmp in HV_K]
            HV_K = np.array(HV_K)
            HV_K = HV_K / (coodinates.reshape(-1,1,1))
            
            hot_fea = np.vstack([make_hot_for_atom_i(atoms,i,HV_K,np.unique(neighbor_list.get_neighbors(i)[0])) for i in range(len(atoms))])
            hot_fea = torch.from_numpy(hot_fea).float()
            hot_fea = torch.Tensor(hot_fea)
        else:
            hot_fea = torch.zeros(atom_fea.shape[0], 1)

        nbr_fea_idx, nbr_fea, site_idx = [], [], []
        distances=atoms.get_all_distances()
        for i in range(len(atoms)):
            nbr = neighbor_list.get_neighbors(i)[0]
            nbr = np.unique(nbr)
            if i in ads_idx:
                site_idx.append(nbr)
            if len(nbr) < self.max_num_nbr:
                #warnings.warn('{} not find enough neighbors to build graph. '
                #              'If it happens frequently, consider increase '
                #              'radius.'.format(cif_id))
                nbr_fea_idx.append(nbr.tolist() +
                                   [i] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(distances[i][nbr].tolist() +
                               [np.nan] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr=nbr[np.argsort(distances[i][nbr])]
                nbr_fea_idx.append(nbr[:self.max_num_nbr].tolist())
                nbr_fea.append(distances[i][nbr[:self.max_num_nbr]])
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.ctf.expand(nbr_fea,atoms,nbr_fea_idx)
        site_idx = np.concatenate(site_idx)
        site_idx = np.unique(site_idx)
        if len(site_idx) == 0:
            site_idx = np.array([0])

        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        site_idx = torch.LongTensor(site_idx)
        target = torch.Tensor([float(target)])
        target1 = torch.Tensor(target1)
        target2 = torch.Tensor([float(target2)])
        atoms_idx = torch.LongTensor(np.arange(len(atoms)))

        return ([atom_fea, hot_fea], nbr_fea, nbr_fea_idx, atoms_idx, site_idx), target, target1, target2, cif_id