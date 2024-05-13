# Pontryagin Neural Operator for Solving General-Sum Differential Games with Parametric State Constraints
<br>
Lei Zhang,
Mukesh Ghimire, 
Zhe Xu, 
Wenlong Zhang, 
Yi Ren<br>
Arizona State University

This is our L4DC paper: <a href="https://arxiv.org/pdf/2401.01502"> "Pontryagin Neural Operator for Solving General-Sum Differential Games with Parametric State Constraints"</a>

## Get started
There exists two different environment, you can set up a conda environment with all dependencies like so:

For Uncontrolled_intersection_complete_information_game
```
conda env create -f environment.yml
conda activate siren
```
For BVP_generation
```
conda env create -f environment.yml
conda activate hji
```

## Code structure
There are two folders with different functions
### BVP_generation: use standard BVP solver to collect the Nash equilibrial (NE) values for uncontrolled intersection
The code is organized as follows:
* `generate_intersection.py`: generate 5D NE values functions under 25 player type configurations.
* `./utilities/BVP_solver.py`: BVP solver.
* `./example/vehicle/problem_def_intersection.py`: dynamic, PMP equation setting for uncontrolled intersection case.

run `generate_intersection.py`, to collect NE values. Please notice there is four player types in case 1. You should give setting in `generate_intersection.py`. Data size can be set in `./example/vehicle/problem_def_intersection.py`.
