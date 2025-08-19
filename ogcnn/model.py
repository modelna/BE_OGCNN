from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Standard Convolutional operation on graphs (BE-OGCNN implementation).
    Updates atom feature representations by extracting structural information 
    from their neighboring nodes.

    Attributes
    ----------
    atom_fea_len: int
        Number of atom hidden features.
    nbr_fea_len: int
        Number of bond features.
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass computing atom neighbor interactions.

        N: Total number of atoms in the batch
        M: Max number of neighbors for each atom

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution processing

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class ImprovedConvLayer(nn.Module):
    """
    Advanced Convolutional operation on graphs with explicit edge 
    updates and both 2-body and 3-body spatial correlations.

    Attributes
    ----------
    atom_fea_len: int
        Number of atom hidden features.
    nbr_fea_len: int
        Number of bond features to process 2-body logic over.
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ImprovedConvLayer setting up edge tracking.
        """
        super(ImprovedConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        
        # Layers for 2-body atom feature updates (original)
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()
        
        # Layers for 2-body edge feature updates
        self.fc_edge = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.nbr_fea_len)
        self.sigmoid_edge = nn.Sigmoid()
        self.softplus_edge = nn.Softplus()
        self.bn_edge = nn.BatchNorm1d(2 * self.nbr_fea_len)

        # --- NEW: Layers for 3-body atom feature updates ---
        # This part is added to incorporate 3-body correlations into the atom update.
        self.fc_3body = nn.Linear(3 * self.atom_fea_len + 2 * self.nbr_fea_len,
                                  2 * self.atom_fea_len)
        self.sigmoid_3body = nn.Sigmoid()
        self.softplus_3body = nn.Softplus()
        self.bn_3body = nn.BatchNorm1d(2 * self.atom_fea_len)
        # --- End of NEW ---

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass with advanced topology processing.

        Parameters
        ----------
        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Current representations of atoms
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Current bond/edge characteristics
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Map array holding neighbor relations
        
        Returns
        -------
        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom features updated globally
        nbr_out_fea: nn.Variable shape (N, M, nbr_fea_len)
          Edge features explicitly updated by adjacent node state shifts
        """
        N, M = nbr_fea_idx.shape
        # Get neighbor atom features
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        
        # --- 2-body interactions ---
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)

        # Atom Feature Update (2-body part)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        two_body_summed = torch.sum(nbr_filter * nbr_core, dim=1)

        # Edge Feature Update (2-body part)
        edge_gated_fea = self.fc_edge(total_nbr_fea)
        edge_gated_fea = self.bn_edge(edge_gated_fea.view(
            -1, self.nbr_fea_len*2)).view(N, M, self.nbr_fea_len*2)
        edge_filter, edge_core = edge_gated_fea.chunk(2, dim=2)
        edge_filter = self.sigmoid_edge(edge_filter)
        edge_core = self.softplus_edge(edge_core)
        nbr_out_fea = nbr_fea + edge_filter * edge_core
        
        # --- NEW: 3-body interactions for atom update ---
        # Expand features to create triplets (i, j, l)
        # atom_in_fea (i) -> (N, 1, 1, atom_fea_len)
        # atom_nbr_fea (j and l) -> (N, M, M, atom_fea_len)
        # nbr_fea (bonds ij and il) -> (N, M, M, nbr_fea_len)
        atom_in_fea_expanded = atom_in_fea.unsqueeze(1).unsqueeze(1).expand(N, M, M, self.atom_fea_len)
        atom_nbr_fea_j = atom_nbr_fea.unsqueeze(2).expand(N, M, M, self.atom_fea_len)
        atom_nbr_fea_l = atom_nbr_fea.unsqueeze(1).expand(N, M, M, self.atom_fea_len)
        nbr_fea_ij = nbr_fea.unsqueeze(2).expand(N, M, M, self.nbr_fea_len)
        nbr_fea_il = nbr_fea.unsqueeze(1).expand(N, M, M, self.nbr_fea_len)

        # Concatenate features for the 3-body message
        total_3body_fea = torch.cat([atom_in_fea_expanded, atom_nbr_fea_j, atom_nbr_fea_l,
                                     nbr_fea_ij, nbr_fea_il], dim=3)
        
        # 3-body gated activation
        three_body_gated_fea = self.fc_3body(total_3body_fea)
        three_body_gated_fea = self.bn_3body(three_body_gated_fea.view(
            -1, self.atom_fea_len * 2)).view(N, M, M, self.atom_fea_len * 2)
        three_body_filter, three_body_core = three_body_gated_fea.chunk(2, dim=3)
        three_body_filter = self.sigmoid_3body(three_body_filter)
        three_body_core = self.softplus_3body(three_body_core)
        
        # Sum over all pairs of neighbors (j, l)
        three_body_summed = torch.sum(three_body_filter * three_body_core, dim=(1, 2))
        # --- End of NEW ---
        
        # --- Final Atom Feature Update ---
        # Combine 2-body and 3-body contributions
        total_summed = self.bn2(two_body_summed + three_body_summed)
        atom_out_fea = self.softplus2(atom_in_fea + total_summed)
        
        return atom_out_fea, nbr_out_fea
    

class OrbitalCrystalGraphConvNet(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len, orig_hot_fea_len,
                 atom_fea_len, hot_fea_len, h_fea_len, n_conv=3, n_h=1, orbital=False, improved=False,
                 classification=False):
        super(OrbitalCrystalGraphConvNet, self).__init__()
        self.orbital = orbital
        self.improved = improved
        self.classification = classification
        if self.orbital:
            self.embedding1 = nn.Linear(orig_atom_fea_len, hot_fea_len)
            self.relu = nn.Softplus()
            self.embedding2 = nn.Linear(hot_fea_len, atom_fea_len)
        else:
            self.embedding1 = nn.Linear(orig_atom_fea_len, atom_fea_len)
        if self.improved:
            self.convs = nn.ModuleList([ImprovedConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        else:
            self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc1 = nn.Linear(atom_fea_len, h_fea_len)

        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out_d = nn.Linear(h_fea_len, 1)
            self.fc_out_e = nn.Linear(h_fea_len, 1)
            self.fc_out1 = nn.Linear(h_fea_len, 1)
            self.fc_out2 = nn.Linear(h_fea_len, 1)
            self.fc_out3 = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, atoms_idx, site_idx, return_feature=False):
        atom_fea = self.embedding1(atom_fea)
        if self.orbital:
            atom_fea = self.relu(atom_fea)
            atom_fea = self.embedding2(atom_fea)
        if return_feature:
            return self.pooling(atom_fea,atoms_idx)
        for conv_func in self.convs:
            if self.improved:
                atom_fea, nbr_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
            else:
                atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        atom_fea = self.conv_to_fc1(self.conv_to_fc_softplus(atom_fea))
        atom_fea = self.conv_to_fc_softplus(atom_fea)   
        if self.classification:
            atom_fea = self.dropout(atom_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                atom_fea = softplus(fc(atom_fea))
        ad_fea = self.pooling(atom_fea, site_idx)

        site_counts = torch.tensor([len(idx_map) for idx_map in site_idx], device=ad_fea.device)
        ad = torch.zeros(ad_fea.shape[0], 1, device=ad_fea.device)
        indices_1 = torch.where(site_counts == 1)[0]
        indices_2 = torch.where(site_counts == 2)[0]
        indices_3_or_more = torch.where(site_counts >= 3)[0]
        if len(indices_1) > 0:
            ad[indices_1] = self.fc_out1(ad_fea[indices_1])
        if len(indices_2) > 0:
            ad[indices_2] = self.fc_out2(ad_fea[indices_2])
        if len(indices_3_or_more) > 0:
            ad[indices_3_or_more] = self.fc_out3(ad_fea[indices_3_or_more])
        out = ad

        out1 = self.fc_out_d(atom_fea)
        out2 = self.fc_out_e(self.pooling(atom_fea, atoms_idx))
        if self.classification:
            out = self.logsoftmax(out)
        return out, out1, out2

    def pooling(self, atom_fea, crystal_atom_idx):
        # assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==  atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)

