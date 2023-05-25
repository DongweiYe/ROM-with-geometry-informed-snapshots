### Non-intrusive reduced-order modeling with geometry-informed snapshots

This code is associated to the publication of "[Data-driven reduced-order modelling for blood flow simulations with geometry-informed snapshots](https://arxiv.org/abs/2302.11006)". 

This repository present the implementation of two synthetic hemodynamics examples, stenosis and bifurcation in 2D. The implementation of the finite element simulation is based on open source [FreeFEM](https://freefem.org/) and the surface registration is acheived by shape analysis software [deformatrica](https://www.deformetrica.org/). The non-instruive reduced order models the examples are constructed with proper orthogonal decomposition (POD) with radial basis function (RBF) interpolation.

The repository contains all the necessary information/data to reproduce the results. The detailed instructions are listed in the following section.

## Perform ROM with geometry-informed snapshots
Each example consists of three main procedures: preprocessing, finite element simulations and reduced-order modelling. Each steps corresponding to a .py/.edp file:
* 'preprocess.py' contains following functions:
  1. Generate samples for geometry variaty         
  2. Shape and mesh generation based on samples (use file: MeshGeneration.edp & ReferMeshGeneration.edp)
  3. Fetch boundary vertice of meshes 
  4. Surface registration (use fold: deformetrica_script, implemented via deformatrica)
  5. Compute the mapping using RBF interpolation and save the data for FEM simulation
  6. Parametrize the geometry and compute reduced parameters of geometry

* 'ReferNS.edp' is designed for performing finite-element simulations with various shapes of domains on a reference domain. This step generate the geometry-informed snapshots for ROM.

* 'ROM.py' collects the snapshots data and constructs the ROM based on POD+RBF interpolation. The script also shows the prediction error on validation and test dataset. The visualization results are saved in 'data/error' directory in the form of .vtk. 
