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
    - The toolsets used to solve 1d Poisson Equation. It defines the truth poisson solution, the PINN class, the functionalities to create discretized data points. It also defines various kernel matrix for natural gradient descent, (L2, H1, H-1, H1 semi, H-1 semi, Fisher-Rao, W2), which are used in the newton-cg optimization framework. (This part will be covered more in detail in the documentation.ipynb file).


