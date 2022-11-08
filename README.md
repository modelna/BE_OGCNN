# Orbital-Graph-Convolutional-Neural-Network
OGCNN
This is the repository for our work on property prediction for crystals. In this work we have used ideas from the Orbital Field matrix and Crystal Graph Convolutional Neural Network to predict material properties with a higher accuracy.

Paper link:https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.4.093801

# Important paper referenced 
The two important papers referenced for this work are:
1. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.145301)
2. Machine learning reveals orbital interaction in crystalline materials,  Science and Technology of Advanced Materials
Volume 18, 2017 - Issue 1.(https://www.tandfonline.com/doi/full/10.1080/14686996.2017.1378060)

We used the ideas from these papers and did some of our modifications to develop the OGCNN which gives a higher performance than the seminal work of CGCNN

# Prerequisites
To run the OGCNN code the following packages are required
- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org). It is preferable to install this package via pip
- [ase](https://wiki.fysik.dtu.dk/ase/)

It is advised to create a new conda environment and then install these packages. To create a new environment please refer to the conda documentation on managing environments (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

## Usage

To input crystal structures to OGCNN, you will need to define a customized dataset. Note that this is required for both training and predicting. 

The dataset that we use for this work are in the cif format. 

- [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) files recording the structure of the crystals that you are interested in
- The values of the target properties for each crystal in the dataset.

You can create a customized pre-training dataset by creating a directory `root_dir` with the following files: 

<!-- 1. `id_prop.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column recodes the value of target property. If you want to predict material properties with `predict.py`, you can put any number in the second column. (The second column is still needed.) -->

1. `atom_init.json`: a [JSON](https://en.wikipedia.org/wiki/JSON) file that stores the initialization vector for each element. An example of `atom_init.json` is `data/sample-regression/atom_init.json`, which should be good for most applications. The `atom_init.json` file has some of the basic atomic features encoded. Please refer the supplementary information of the paper to find out more about the basic atomic features.

2. `ID.cif`: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

The structure of the `root_dir` should be:

```
root_dir
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```
In this work, we aggregated the Matminer and the hMOF database for pretraining. The SSL models are likely to train better with larger datsets, if the users have aggregated larger datasets they can add it to the  `root_dir` 


