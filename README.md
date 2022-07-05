# Joint Multidimensional Scaling
 
Joint MDS is a an approach for unsupervised manifold alignment, which maps datasets from two different domains without any known correspondences between data instances across the datasets, to a common low-dimensional Euclidean space. Joint MDS integrates Multidimensional Scaling (MDS) and Wasserstein Procrustes analysis into a joint optimization problem to simultaneously generate isometric embeddings of data and learn correspondences between instances from two different datasets, while only requiring intra-dataset pairwise dissimilarities as input.

## Installation

The dependencies are managed by [miniconda](https://conda.io/miniconda.html)

```
python=3.9
numpy
scipy
pytorch=1.9.1
scikit-learn
pandas
```

## Usage

The usage of Joint MDS is similar to the [MDS](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html) function in scikit-learn.
Here is one minimal example of Joint MDS.


```
from joint_mds import joint_mds
import numpy as np

D1 = np.random.rand(128, 10)
D2 = np.random.rand(64, 20）

JMDS = joint_mds(n_components=2, dissimilarity="eculidean")
Z1, Z2, P = JMDS.fit_transform(D1, D2)

print(Z1.shape) # (128, 2)
print(Z2.shape) # (64, 2)
print(P.shape)  # (128, 64)
```

For all paramters in Joint MDS please refer to `joint_mds.py`.

The folder `examples` contains scripts to demonstrate the use of Joint MDS for 
different tasks, including unsupervised heterogeneous domain adaptation, 
graph matching, protein structure alignment.

## Citation
If you find this repository useful in your research, 
please consider citing the following paper:


## Contact
[dexiong.chen@bsse.ethz.ch](mailto:dexiong.chen@bsse.ethz.ch) or 
[bowen.fan@bsse.ethz.ch](mailto:bowen.fan@bsse.ethz.ch)