Firstly please note, that for sure, the docs and tutorials of all used packages were used as a source of information to create the code. So thanks to all creators for the effort of creating the docs and tutorials of their package!

# PredBind
Prediction of Binding Affinities using neuronal networks. This package was developed during my master-thesis. The topic was the prediction of binding affinities using SchNet. 
* Schütt, Kristof T., et al. "SchNet–A deep learning architecture for molecules and materials." 
  The Journal of Chemical Physics148.24 (2018): 241722.

The starting point was the work of Jiménez, et.al. "KDEEP: protein–ligand absolute binding affinity prediction via 3D-convolutional neural networks".

The objective of this work was, to reproduce the results of KDeep and check, weather a SchNet can outperform a CNN in the prediction of binding affinitied. The idea was, that SchNet is rotational invariant, so no data augmentation is necessary.
Nevertheless, the result was, that SchNet seems not able to predict binding affinities. The assumption is, that the complexity of SchNet is too high for the PDBbind-dataset. This dataset was used as training and benchmark set and "only" contains 4444 data points. 

Nevertheless, to understand what I have done, this package was published. It provides the whole code, which was produced during the thesis. The usage is explained in the thesis and also in the scripts. Feel free, to play with it. If you find bugs or have questions, please open a ticket.

See package is divided in two parts.

# CNN -> KDeep reproduction 
* Jiménez, J., Skalic, M., Martinez-Rosell, G., & De Fabritiis, G. (2018). K DEEP: protein–ligand absolute binding affinity 
  prediction via 3D-convolutional neural networks. Journal of chemical information and modeling, 58(2), 287-296.

You will need moleculekit/htmd.
* HTMD: High-Throughput Molecular Dynamics for Molecular Discovery S. Doerr, M. J. Harvey, Frank Noé, and G. De Fabritiis           
  Journal of Chemical Theory and Computation 2016 12 (4), 1845-1852
Also PyTorch, h5py, os, pybel, csv, numpy, matplotlib and pandas is necessary, to use all methods in this work. All but PyTorch are preinstalled in common, if you use Anaconda.

# Schnet
The SchNet-Part uses a modified version of schnetpack which can be found at the modified fork on my profile -> https://github.com/BioFreak95/schnetpack 
Maybe these changes can be added to the original fork in future.

* Schütt, K. T., et al. "SchNetPack: A deep learning toolbox for atomistic systems." 
  Journal of chemical theory and computation 15.1 (2018): 448-455.

To install, download the fork and use `pip install .` in the directory of schnetpack.

Next to this, following packages are necessary:
datatime, PyTorch, matplotlib, pandas, scipy, numpy, sys, h5py, os, csv and ase. 
If you want to use the same features, as used in the KDeep-paper, you will also need HTMD, which was used to calculate them.
