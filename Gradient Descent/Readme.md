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

### 3. Initialize Parameters

```matlab
x = [5; 5];  % Starting point
tol = 1e-6;  % Tolerance for stopping
max_iter = 1000;  % Maximum number of iterations
alpha = 1e-3;  % Initial step size
```

- **`x`**: Initial point of the descent process.
- **`tol`**: Convergence threshold; the algorithm stops if the gradient norm is below this value.
- **`max_iter`**: Maximum allowable iterations to avoid infinite loops.
- **`alpha`**: Initial step size for line search.

### 4. Storing the Path

```matlab
path = x';
```

Tracks all points visited during the optimization for visualization.

### 5. Gradient Descent Loop

```matlab
for iter = 1:max_iter
    grad = grad_f(x);  % Calculate the gradient
    grad_norm = norm(grad);  % Norm of the gradient
    
    if grad_norm < tol  % Convergence check
        fprintf('Converged in %d iterations\n', iter);
        break;
    end
    
    step_size = alpha;
    while f(x - step_size * grad) > f(x) - 0.5 * step_size * grad_norm^2
        step_size = step_size * 0.5;  % Backtracking line search
    end
    
    x = x - step_size * grad;  % Update x
    path = [path; x'];  % Store the point
end
```

- **Gradient Calculation**: Computes the gradient at the current position.
- **Convergence Check**: Stops the algorithm if the gradient norm is below `tol`.
- **Line Search**: Adjusts the step size \( \alpha \) to ensure sufficient decrease in \( f(x) \).
- **Update Rule**: Moves in the direction \( -\nabla f(x) \).

### 6. Final Outputs

```matlab
fprintf('Optimal point: (%.4f, %.4f)\n', x(1), x(2));
fprintf('Objective function value at optimal point: %.4f\n', f(x));
```

Displays the optimized point and the value of the function at this point.

### 7. Visualization

#### Contour Plot and Optimization Path

```matlab
figure;
hold on;

[X, Y] = meshgrid(-5:0.1:5, -5:0.1:5);
Z = arrayfun(@(x, y) f([x; y]), X, Y);
contour(X, Y, Z, 50, 'LineWidth', 1);  % Contour plot
plot(path(:,1), path(:,2), 'ro-', 'LineWidth', 1, 'MarkerSize', 2, 'MarkerFaceColor', 'r');  % Path

text(path(1,1), path(1,2), 'Start', 'FontSize', 10, 'FontWeight', 'bold');
text(path(end,1), path(end,2), 'End', 'FontSize', 10, 'FontWeight', 'bold');
xlabel('x_1');
ylabel('x_2');
grid on;
title('Fastest Descent Optimization Path');
```

This plot shows:
- Contours of the function \( f(x) \).
- The optimization path traced by the algorithm.

#### Surface Plot of \( f(x) \)

```matlab
figure;
surf(X, Y, Z, 'EdgeColor', 'none');
title('Surface Plot of Objective Function f(x)');
xlabel('x_1');
ylabel('x_2');
zlabel('f(x)');
grid on;
colormap(jet);
colorbar;
```

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
