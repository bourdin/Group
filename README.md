# Shallow Water Equation
 
 These files are Python code files created to explore the 1D shallow water equation. The files consist of the following three files:
 
 * Function.py
 
 * SWE.py
 
 * main.py
 
 Function.py is a library created to define functions used for differentiation and approximation, while SWE is a class that defines various functions needed for the experiments. Finally, main contains a collection of example code.

 Python version should be more than 3.10 to run the 'match' and 'case' in the code. And the following libraries are necessary to run the code.
 * numpy
 * 
 
 Updates are planned for the future.
 
 
 ## Function.py
 
 Function.py is built-in function, which includes interpolation and differential method which is used in SWE class. The belows are the contents summary.

  ### Function for the interpolation
  We're goint to use cell just like grid. We assume that the perturvation of wave (eta and h) is loacted in the center of the cell and velocity (u) is located in the edge of cell.
  To interpolate each other, we need interploration function to interpolate note to edge and edge to node.
   * interpolation_node_to_edge(eta, BoundaryCondition)
   * interpolation_edge_to_node(u, BoundaryCondition)
  These two function is used for interpolation. We choose to use linear interpolation to impliment.
   * linear_interpolation(v1,v2)
  This function is used to choose the way to implement. It works for actual calculation.

  ### 1D version dirivative calculation
  It all includes the following dirivative calculation.
   * divergence
   * grad (this is for graident)
   * Laplacian
   * derivative

  ### Flux and Bernoulli Function
  Flux and Bernoulli function can be calculated with different boundary condtion.
   * Flux
   * Bernoulli
 
 
 ## SWE.py
 
 SWE is the class for testing shallow water equation. You can make a new test set with this simple code: test=SWE().
