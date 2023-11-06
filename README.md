# GeoIPD
Inexact Penalty Decomposition Methods for Optimization Problems with Geometric Constraints

### General Information
Code related to the paper

[Kanzow, Christian, and Matteo Lapucci. "Inexact penalty decomposition methods for optimization problems with geometric constraints." Computational Optimization and Applications (2023): 1-35.](https://link.springer.com/article/10.1007/s10589-023-00475-2)

If you have used our code for research purposes, please cite the publication mentioned above.
For the sake of simplicity, we provide the Bibtex format:

```
@article{kanzow2023inexact,
  title={Inexact penalty decomposition methods for optimization problems with geometric constraints},
  author={Kanzow, Christian and Lapucci, Matteo},
  journal={Computational Optimization and Applications},
  pages={1--35},
  year={2023},
  publisher={Springer}
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

