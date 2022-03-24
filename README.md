# Understanding the Difficulty of Training Physics-Informed Neural Networks on Dynamical Systems

Physics-informed neural networks (PINNs) seamlessly integrate data and physical constraints into the solving of problems governed by differential equations.
In settings with little labeled training data, their optimization relies on the complexity of the embedded physics loss function.
Two fundamental questions arise in any discussion of frequently reported convergence issues in PINNs:
Why does the optimization often converge to solutions that lack physical behavior?
And why do reduced domain methods improve convergence behavior in PINNs?
We answer these questions by studying the physics loss function in the vicinity of fixed points of dynamical systems.
Experiments on a simple dynamical system demonstrate that solutions corresponding to nonphysical system dynamics can be dominant in the physics loss landscape and optimization.
We find that reducing the computational domain lowers the optimization complexity and chance of getting trapped with nonphysical solutions.

## Note

This TF2 code provides the basic PINN framework for reproducing our results on a simple autonomous differential equation. 
