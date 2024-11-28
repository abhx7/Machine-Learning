% Define the objective function
f = @(x) x(1)^2 + 3*x(2)^2 + 2*x(1)*x(2);  % Example: f(x) = x1^2 + 3*x2^2 + 2*x1*x2

% Define the gradient of the objective function
grad_f = @(x) [2*x(1) + 2*x(2); 6*x(2) + 2*x(1)];

% Initialize parameters
x = [5; 5];  % Starting point
tol = 1e-6;  % Tolerance for stopping
max_iter = 1000;  % Maximum number of iterations
alpha = 1e-3;  % Initial step size

% To store the path
path = x';

% Fastest Descent Loop
for iter = 1:max_iter
    grad = grad_f(x);  % Calculate the gradient at current x
    grad_norm = norm(grad);  % Norm of the gradient for convergence check

    % Check for convergence
    if grad_norm < tol
        fprintf('Converged in %d iterations\n', iter);
        break;
    end
    
    % Line search for step size (simple backtracking)
    step_size = alpha;
    while f(x - step_size * grad) > f(x) - 0.5 * step_size * grad_norm^2
        step_size = step_size * 0.5;
    end
    
    % Update rule for x
    x = x - step_size * grad;
    path = [path; x'];  % Store the updated point in the path
end

% Final output
fprintf('Optimal point: (%.4f, %.4f)\n', x(1), x(2));
fprintf('Objective function value at optimal point: %.4f\n', f(x));

% Visualization
figure;
hold on;

% Define the grid for contour plot
[X, Y] = meshgrid(-5:0.1:5, -5:0.1:5);
Z = arrayfun(@(x, y) f([x; y]), X, Y);

% Plot the contour of the function
contour(X, Y, Z, 50, 'LineWidth', 1);
colormap(jet);
colorbar;
caxis([0, 30]);
title('Fastest Descent Optimization Path');
xlabel('x_1');
ylabel('x_2');

% Plot the path of descent
plot(path(:,1), path(:,2), 'ro-', 'LineWidth', 1, 'MarkerSize', 2, 'MarkerFaceColor', 'r');
text(path(1,1), path(1,2), 'Start', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', 'FontSize', 10, 'FontWeight', 'bold');
text(path(end,1), path(end,2), 'End', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', 'FontSize', 10, 'FontWeight', 'bold');

% Formatting the plot
grid on;
axis equal;
legend({'Contour of f(x)', 'Path of Descent'}, 'Location', 'best');
set(gca, 'FontSize', 12, 'FontWeight', 'bold');

hold off;

% Surface plot of the function
figure;
surf(X, Y, Z, 'EdgeColor', 'none');
colormap(jet);
colorbar;
title('Surface Plot of Objective Function f(x)');
xlabel('x_1');
ylabel('x_2');
zlabel('f(x)');
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
grid on;

