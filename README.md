## PINN to Solver Poisson Equation

In this repository, we use physically imposed neural network to solve Poisson equations.

## Repository Structure 

- PINN_tool.py
- PINN2D_tool.py
- documentation.ipynb
- README.md
- optimizer
    - optimizer.py
- 1D_poisson
    - GD_1dPoisson.py
    - GaussNewton_1dPoisson.py
    - H1_1dPoisson.py
    - H1semi_1dPoisson.py
    - Hinv_1dPoisson.py
    - Hinvsemi_1dPoisson.py
- 2D_poisson
    - GD_2dPoisson.py
    - GaussNewton_2dPoisson.py
    - H1_2dPoisson.py
    - H1semi_2dPoisson.py
    - Hinv_2dPoisson.py
    - Hinvsemi_2dPoisson.py
    - FR_2dPoisson.py
    - BFGS_2dPoisson.py
    - W2_2dPoisson.py
- mini_batch
    - GD_2dPoisson_mini.py
    - GaussNewton_2dPoisson_mini.py
    - H1_2dPoisson_mini.py
    - H1semi_2dPoisson_mini.py
    - Hinv_2dPoisson_mini.py
    - Hinvsemi_2dPoisson_mini.py
    - FR_2dPoisson_mini.py

## Files and Repository Description
- documentation.ipynb
    - All detailed documentations of all methods and experiments will be here, including discretization methods, optimization framework. 
- PINN_tool.py
    - The toolsets used to solve 1d Poisson Equation. It defines the truth poisson solution, the PINN class, the functionalities to create discretized data points. It also defines various kernel matrix for natural gradient descent, (L2, H1, H-1, H1 semi, H-1 semi, Fisher-Rao, W2), which are used in the newton-cg optimization framework. (This part will be covered more in detail in the documentation.ipynb file)
- PINN2D_tool.py
    - The toolsets used to solve 2d Poisson Equation. It defines the truth poisson solution, the PINN class, the functionalities to create discretized data points. It also defines varius kernel matrix for natural gradient descent, (L2, H1, H-1, H1 semi, H-1 semi, Fisher-Rao, W2), which are used in the newton-cg optimization framework. (This part will be covered more in detail in the documentation.ipynb file)

- 1D_poisson
    - This folder contains all the experiments of using PINN to solve 1D poisson equation.
- GD_1dPoisson.py
    - This file is the experiment of using standard gradient descent to solve the 1d poisson equation using PINN.
- GaussNewton_1dPoisson.py
    - This file is the experiment of using Gauss Newton method (L2 natural gradient desent) to solve the 1d Poisson equation using PINN.
- H1_1dPoisson.py
    - This file is the experiment of using H1 natrual gradient descent to solve the 1d Poisson equation using PINN.
- H1semi_1dPoisson.py
    - This file is the experiment of using H1 seminorm natural gradient descent to solve the 1d Poisson equation using PINN.
- Hinv_1dPoisson.py
    - This file is the experiment of using H-1 norm natural gradient descent to solve the 1d Poisson equation using PINN.
- Hinvsemi_1dPoisson.py
    - This file is the experiment of using H-1 seminorm natural gradient descent to solve the 1d Poisson equation using PINN.

- 2D_poisson
    - This folder contains all the experiments of using PINN to solve 2D poisson equation.
- GD_2dPoisson.py
    - This file is the experiment of using standard gradient descent to solve the 2d poisson equation using PINN.
- GaussNewton_2dPoisson.py
    - This file is the experiment of using Gauss Newton method (L2 natural gradient desent) to solve the 2d Poisson equation using PINN.
- H1_2dPoisson.py
    - This file is the experiment of using H1 natrual gradient descent to solve the 2d Poisson equation using PINN.
- H1semi_2dPoisson.py
    - This file is the experiment of using H1 seminorm natural gradient descent to solve the 2d Poisson equation using PINN.
- Hinv_2dPoisson.py
    - This file is the experiment of using H-1 norm natural gradient descent to solve the 2d Poisson equation using PINN.
- Hinvsemi_2dPoisson.py
    - This file is the experiment of using H-1 seminorm natural gradient descent to solve the 2d Poisson equation using PINN.
- FR_2dPoisson.py
    - This file is the experiment of using Fisher-Rao natural gradient descent to solve the 2d Poisson equation using PINN.
- BFGS_2dPoisson.py
    - This file is the experiment of using L-BFGS-B method to solve the 2d Poisson equation using PINN.
- W2_2dPoisson.py
    - This file is the experiment of using W2 (Wasserstein) natural gradient descent to solve the 2d Poisson equation using PINN.

- optimizer
    - Folder that contains the code for mini batch optimizer
- optimizer.py
    - Module that contains all mini-batch optimizer, including the mini-batch version of std gradient descent, L2, H1, H-1, H1 semi, H-1 semi, Fisher-Rao natural gradient descent.

- mini_batch
    - Folder that contains all the experiments of mini-batch GD optimization of the PINN model solving 2d poisson equation.
- GD_2dPoisson_mini.py
    - Experiment of mini-batch version of standard gradient descent of PINN solving 2d Poisson equation.
- GaussNewton_2dPoisson_mini.py
    - Experiment of mini-batch version of L2 natural gradient descent of PINN solving 2d Poisson equation.
- H1_2dPoisson_mini.py
    - Experiment of mini-batch version of H1 natural gradient descent of PINN solving 2d Poisson equation.
- H1semi_2dPoisson_mini.py
    - Experiment of mini-batch version of H1 seminorm gradient descent of PINN solving 2d Poisson equation.
- Hinv_2dPoisson_mini.py
    - Experiment of mini-batch version of H-1 norm naural gradient descent of PINN solving 2d Poisson equation.
- Hinvsemi_2dPoisson_mini.py
    - Experiment of mini-batch version of H-1 seminorm gradient descent of PINN solving 2d Poisson equation.
- FR_2dPoisson_mini.py
    - Experiment of mini-batch version of Fisher-Rao natural gradient descent of PINN solving 2d Poisson equation.
