![](https://github.com/DongweiYe/ROM-with-geometry-informed-snapshots/blob/main/github_figure.png)
### Non-intrusive reduced-order modelling with geometry-informed snapshots
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10029572.svg)](https://doi.org/10.5281/zenodo.10029572)

This repository is associated to the publication of "[Data-driven reduced-order modelling for blood flow simulations with geometry-informed snapshots](https://arxiv.org/abs/2302.11006)". 

This repository presents the implementation of two synthetic hemodynamics examples, stenosis and bifurcation in 2D. The implementation of the finite element simulation is based on open source [FreeFEM](https://freefem.org/) and the surface registration is achieved by shape analysis software [deformatrica](https://www.deformetrica.org/). The non-intrusive reduced order models are constructed with proper orthogonal decomposition (POD) with radial basis function (RBF) interpolation.

The repository contains all the necessary information/data to reproduce the results. The detailed instructions are listed in the following section. The best environment would be Python 3.7 because that is the latest supported version of deformetrica and you may need manually drop out the unnecessary `.decode('utf-8')` in `/your-directory-to-package/python3.7/site-packages/deformetrica/core/estimators/scipy_optimize.py`.

### Perform ROM with geometry-informed snapshots
Each example consists of three main procedures: preprocessing, finite element simulations and reduced-order modelling. Each step corresponds to a .py/.edp file:
* `preprocess.py` contains the following functions (Example 2 is slightly different, check the comments in the corresponding file for details):
  - Generate samples for geometry variety         
  - Shape and mesh generation based on samples (use file: MeshGeneration.edp & ReferMeshGeneration.edp)
  - Fetch boundary vertice of meshes 
  - Surface registration (use folder: deformetrica_script, implemented via deformatrica)
  - Compute the mapping using RBF interpolation and save the data for FEM simulation
  - Parametrize the geometry and compute reduced parameters of geometry

* We provide three .edp here:
  - `ReferNS_steady_parallel.edp` is designed for performing finite-element simulations (steady) with various shapes of domains on a reference domain (parallelized with PETSc). This step generates the geometry-informed snapshots for ROM (recommend to use). 
    Implement with `ff-mpirun -np 16 ReferNS_steady_parallel.edp -v 0` where 16 stands for the number of cores for mpi.
  - `ReferNS_transient_parallel.edp` is the transient version. It takes much more time to generate a steady solution, but can be used for time-dependent problems.
  - `StandardNS.edp` is the standard finite-element simulation for the flow on its original domain. It is used for validation of the solution achieved on the reference domain

* `ROM.py` collects the snapshots data and constructs the ROM based on POD+RBF interpolation. The script also shows the prediction error on the validation and test dataset. The visualization results are saved in `data/error` directory in the form of .vtk. 

Note that surface registration and finite elements are computationally expensive. It might take a long time
