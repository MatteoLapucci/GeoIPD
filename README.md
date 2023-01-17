# GeoIPD
Inexact Penalty Decomposition Methods for Optimization Problems with Geometric Constraints

### General Information
Code related to the paper

[Kanzow, C., Lapucci, M. Inexact Penalty Decomposition Methods for Optimization Problems with Geometric Constraints. arxiv. (2022).](https://doi.org/10.48550/arXiv.2210.05379)

If you have used our code for research purposes, please cite the publication mentioned above.
For the sake of simplicity, we provide the Bibtex format:

```
@article{lapucci2022inexact,
  title={Inexact Penalty Decomposition Methods for Optimization Problems with Geometric Constraints},
  author={Lapucci, Matteo and Kanzow, Christian},
  journal={arXiv preprint arXiv:2210.05379},
  year={2022}
}
```

### Installation

In order to execute the code, you need an [Anaconda](https://www.anaconda.com/) environment. We provide YAML file in order to facilitate the installation of the latter.

##### For Linux user

Open a terminal in the project root folder and execute the following command.

```
conda env create -f geoipd.yml
```


#### Main Packages

* ```python v3.9```
* ```numpy v1.22.3```
* ```scipy v1.7.3```
* ```matplotlib v3.5.2```
* ```gurobipy v9.5.2```

#### Gurobi Optimizer

In order to run some parts of the code, the [Gurobi](https://www.gurobi.com/) Optimizer needs to be installed and, in addition, a valid Gurobi licence is required. 
However, the employment of the Gurobi Optimizer is only required for a subset of experiments and test problems. The PD algorithm does not employ Gurobi, nor the implementations of sparisity and low rank constrained problems.

### Usage

Run main.py (command ```python main.py``` in linux terminal) to test the code; comment/uncomment function calls at the end of the file to reproduce the different experiments reported in the paper.

### Contact

If you have any question, feel free to contact me:

[Matteo Lapucci](https://webgol.dinfo.unifi.it/matteo-lapucci/)<br>
Global Optimization Laboratory ([GOL](https://webgol.dinfo.unifi.it/))<br>
University of Florence<br>
Email: matteo dot lapucci at unifi dot it

