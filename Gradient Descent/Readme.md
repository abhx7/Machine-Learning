# Fastest Descent Optimization 

## Basic Problem Explaination (fastest_descent.m) 

## Problem Description

We aim to minimize the function:
\[ f(x) = x_1^2 + 3x_2^2 + 2x_1x_2 \]

### Steps Explained

### 1. Define the Objective Function

```matlab
f = @(x) x(1)^2 + 3*x(2)^2 + 2*x(1)*x(2);
```

This defines the function \( f(x) \), which is convex and has a single minimum, making it suitable for gradient descent.

### 2. Define the Gradient of the Function

```matlab
grad_f = @(x) [2*x(1) + 2*x(2); 6*x(2) + 2*x(1)];
```

The gradient \( \nabla f(x) \) gives the direction of steepest ascent. Gradient descent uses \( -\nabla f(x) \) to move towards the minimum.

### 3. Initialize Parameters and Storing the Path

- **`x`**: Initial point of the descent process.
- **`tol`**: Convergence threshold; the algorithm stops if the gradient norm is below this value.
- **`max_iter`**: Maximum allowable iterations to avoid infinite loops.
- **`alpha`**: Initial step size for line search.

Storing the path so that it tracks all points visited during the optimization for visualization.

### 4. Gradient Descent Loop

- **Gradient Calculation**: Computes the gradient at the current position.
- **Convergence Check**: Stops the algorithm if the gradient norm is below `tol`.
- **Line Search**: Adjusts the step size \( \alpha \) to ensure sufficient decrease in \( f(x) \).
- **Update Rule**: Moves in the direction \( -\nabla f(x) \).

### 5. Final Outputs

Displays the optimized point and the value of the function at this point.

### 6. Visualization

#### Contour Plot and Optimization Path

![Optimization Path](https://github.com/abhx7/Machine-Learning/blob/main/Gradient%20Descent/graddescentpath.png "Optimization Path Visualization")
 
This plot shows:
- Contours of the function \( f(x) \).
- The optimization path traced by the algorithm.

#### Surface Plot of \( f(x) \)

![Objective Function](https://github.com/abhx7/Machine-Learning/blob/main/Gradient%20Descent/objfunction.png "Objective Function")

This 3D surface plot provides a visualization of the function's landscape.

## Results

1. **Optimal Point**: The algorithm converges to the function's minimum, which is displayed in the console.
2. **Descent Path**: The plotted trajectory shows how the algorithm moves from the starting point to the optimal point.
3. **Convergence**: The algorithm stops when the gradient norm becomes smaller than the tolerance.

---

## How to Run

1. Copy the MATLAB script into a `.m` file.
2. Run the script in MATLAB.
3. Observe the outputs in the console and the plots.

## Files

- `gradient_descent.m`: MATLAB script containing the implementation.
- `README.md`: This explanation.

## License

This project is licensed under the MIT License.

---

Happy optimizing!
