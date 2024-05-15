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
* `generate_intersection.py`: generate 5D NE values functions under 25 player type configurations $\Theta^2$.
* `./utilities/BVP_solver.py`: BVP solver.
* `./example/vehicle/problem_def_intersection.py`: dynamic, PMP equation setting for uncontrolled intersection case.

run `generate_intersection.py`, to collect NE values. Please notice there is 25 player types for uncontrolled intersection case. You should give setting in `generate_intersection.py`. Data size can be set in `./example/vehicle/problem_def_intersection.py`.


### Uncontrolled_intersection_complete_information_game: train hybrid neural operator and Pontryagin neural operator to complete saftety performance test for uncontrolled intersection case with complete information
The code is organized as follows:
* `dataio.py`: load training data for HNO and PNO.
* `training_hno.py`: contains HNO training routine.
* `training_pno.py`: contains PNO training routine.
* `action_functions.py`: contains action computed by costate net for PNO.
* `traj_functions.py`: contains forward state trajectory for PNO.
* `bound_functions.py`: contains boundary condition for value and costate for PNO.
* `value_functions.py`: contains backward approximate value trajectory for PNO.
* `sampling_functions.py`: contains evolutionary sampling for PNO.
* `loss_functions.py`: contains loss functions for HNO and PNO.
* `modules_hno.py`: contains HNO architecture.
* `modules_pno.py`: contains PNO architecture.
* `utils.py`: contains utility functions.
* `diff_operators.py`: contains implementations of differential operators.
* `./experiment_scripts/train_intersection_HNO.py`: contains scripts to train the HNO model for uncontrolled intersection case, which can reproduce experiments in the paper.
* `./experiment_scripts/train_intersection_PNO.py`: contains scripts to train the PNO model for uncontrolled intersection case, which can reproduce experiments in the paper.
* `./validation_scripts/closedloop_traj_generation_hno_tanh.py`: use HNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_hno_tanh_sym.py`: use HNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_pno_tanh.py`: use PNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_pno_tanh_sym.py`: use PNO (tanh as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_pno_relu.py`: use PNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_pno_relu_sym.py`: use PNO (relu as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_pno_sine.py`: use PNO (sine as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/closedloop_traj_generation_pno_sine_sym.py`: use PNO (sine as activation function) as closed-loop controllers to generate data for saftety performance.
* `./validation_scripts/trajectory_with_value_nop.py`: visualize saftety performance for value network (tanh, relu and sin as activation function).
* `./validation_scripts/data_interpolation_nop.py`: create finner state interval for the generated trajectories using neurla operator. Compute more accurate collision rates.
* `./validation_scripts/plot_intersection.py`: visualize parameter function $a(\rm x, \theta)$ on the lattices.
* `./validation_scripts/deeponet_value_contour.py`: visualize value contour of the six learning trunk net output.
* `./validation_scripts/deeponet_branch_visualization.py`: visualize sorted sample mean and std of coefficients $|b_k|$ over $\Theta^2$.
* `./validation_scripts/model`: experimental model in the paper.
* `./validation_scripts/train_data`: training data in the paper.
* `./validation_scripts/test_data`: testing data in the paper.
* `./validation_scripts/closed_loop`: store data by using value network as closed-loop controllers, data used for paper is ready. Download the generated trajectories: <a href="https://drive.google.com/drive/folders/1--zWTasWZLNe6PQz1gA2fr9dhDYJ6fsb?usp=sharing"> link.
* `./example/vehicle/problem_def_intersection.py`: dynamic, PMP equation setting for uncontrolled intersection case.

## Contact
If you have any questions, please feel free to email the authors.

