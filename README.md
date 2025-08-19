# Bond Embedding Orbital Graph Convolutional Neural Network (BE-OGCNN)

This repository contains the implementation of the Bond Embedding Orbital Graph Convolutional Neural Network (BE-OGCNN). This framework builds upon the principles of Graph Convolutional Neural Networks and Orbital Field Matrices to accurately predict material properties, particularly focusing on catalytic surfaces and adsorption phenomena. 

By running the script with the `--orbital` flag, you activate the BE-OGCNN architecture described in the accompanying draft. Using the `--improved` flag switches the architecture to BE-ICGCNN (Bond Embedding Improved Crystal Graph Convolutional Neural Network), which includes explicit higher-order edge-updating mechanics for the graphs.

## Important Papers Referenced

The methodology in this repository is heavily inspired and modified from the following works:

1. **BE-CGCNN**: *Bond Embedding Crystal Graph Convolutional Neural Network*
   DOI: [10.1038/s41467-023-38758-1](https://doi.org/10.1038/s41467-023-38758-1)
2. **OGCNN**: *Machine learning reveals orbital interaction in crystalline materials*
   DOI: [10.1103/PhysRevMaterials.4.093801](https://doi.org/10.1103/PhysRevMaterials.4.093801)

## Prerequisites

To run the BE-OGCNN framework, the following packages are required:

- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [ase](https://wiki.fysik.dtu.dk/ase/) (Atomic Simulation Environment)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)

It is highly recommended to create a dedicated conda environment and then install these libraries.

## Data Format and Preparation

Unlike standard CGCNN/OGCNN which often relies on collections of CIF files, **BE-OGCNN utilizes `.xyz` trajectory files or ASE Atoms object lists** directly.

To train or predict using your own structures, prepare an `.xyz` file (e.g., `data.xyz`) where each frame corresponds to a crystal/surface structure.

### Adding Targets to the XYZ file
In order for BE-OGCNN to train effectively on your data, you must embed the target properties directly into the `ase.Atoms` objects before saving the `.xyz` file. 

1. **Adsorption Energy (`ad`)**:
   Add the adsorption energy target to the `.info` dictionary of each atom object:
   ```python
   atoms.info['ad'] = <adsorption_energy_value>
   ```

2. **d-band Center (`d_band_center`)**:
   Add the d-band center target as an array representing values for each atom:
   ```python
   atoms.set_array('d_band_centers', <d_band_center_array>) # Should match the length of atoms
   ```

## Usage

### Training a BE-OGCNN Model

To train a new model, use `main.py` and point it to your `.xyz` file path. 
For example, to train using BE-OGCNN on a sample `.xyz` file:

```bash
python main.py /path/to/your/data.xyz --orbital
```

To run the **BE-ICGCNN** variant (Improved) instead:
```bash
python main.py /path/to/your/data.xyz --orbital --improved
```

#### Adjusting Multi-Task Loss Weights
When dealing with multi-task regression (e.g. predicting Adsorption Energy and d-band centers simultaneously), you can adjust the contribution of each target to the final loss function multipliers using the weight flags:
- `--ad-weight`: Loss weight for the primary Adsorption Energy target
- `--d-weight`: Loss weight for the secondary target (e.g. d-band center)
- `--e-weight`: Loss weight for the tertiary target (e.g. total energy)

Example:
```bash
python main.py /path/to/data.xyz --orbital --ad-weight 1.0 --d-weight 0.5
```

You can define the sizes of the Training, Validation, and Test sets by using the flags `--train-size`, `--val-size`, and `--test-size`.

After training completes, three files are generated in your current working directory:
- `model_best.pth.tar`: Stores the BE-OGCNN model checkpoint possessing the best validation error.
- `checkpoint.pth.tar`: Stores the BE-OGCNN model checkpoint at the final epoch.
- `test_results.csv`: A CSV storing the true values versus predicted values for the test set.

### Making Predictions with a Pre-Trained Model

If you have a previously trained model, you can evaluate new crystals/structures by using the `predict.py` script. You must provide the path to your trained model checkpoint (`.pth.tar`) and the targeted `.xyz` dataset.

```bash
python predict.py /path/to/model_best.pth.tar /path/to/new_data.xyz
```

Predictions will be written to `test_results.csv` in your working directory.
