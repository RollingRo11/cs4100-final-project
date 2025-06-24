# %% cells for my editor!
import numpy as np
import plotly.graph_objects as go

# %% Section 1.1

def gradient_descent(f, deriv_f, x0, alpha, epsilon=1e-5, iter_max=1000):
    """
    Implements gradient descent algorithm

    Args:
        f: function to minimize
        deriv_f: derivative of function
        x0: initial point
        alpha: step size
        epsilon: tolerance for convergence
        iter_max: maximum iterations

    Returns:
        x: optimal point found
        iter: number of iterations performed
    """
    x = x0
    iter = 0

    while iter < iter_max:
        x_new = x - alpha * deriv_f(x)

        if abs(x_new - x) < epsilon:
            break

        x = x_new
        iter += 1

    return x, iter

def f1(x):
    """
    Args:
        x: the inputted value of the function.

    Returns:
        The value of x squared (for function 1)
    """
    return x**2

def deriv_f1(x):
    """
    Args:
        x: the inputted value of the function

    Returns:
        the derivative of the function applied to the input
    """
    return 2*x

def f2(x):
    """
    Args:
        x: the input value to the function

    Returns:
        the quadratic applied to x
    """
    return x**2 - 2*x + 3

def deriv_f2(x):
    """
    Args:
        x: the input value to the derivative

    Returns:
        the derivative of the function applied to x

    """
    return 2*x - 2

def plot_opt(f, x_opt, title):
    """
    Plot a function with the optimal point

    Args:
        f: the function to be plotted
        x_opt: the optimal value of x across the function
        title: the title of the plot
    """
    x = np.linspace(-5, 5, 200)
    y = [f(xi) for xi in x]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Function', line=dict(color='blue')))

    fig.add_trace(go.Scatter(x=[x_opt], y=[f(x_opt)], mode='markers',
                            name=f'Optimal point: x={x_opt:.3f}',
                            marker=dict(color='red', size=12))) # type: ignore

    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title='f(x)',
        showlegend=True,
        hovermode='x'
    )

    fig.show()

def main():
    # params for 1.1
    x0 = 3
    alpha = 0.1
    epsilon = 0.001

    # f1
    x_opt1, iter1 = gradient_descent(f1, deriv_f1, x0, alpha, epsilon)
    plot_opt(f1, x_opt1, "function 1")

    # f2
    x_opt2, iter2 = gradient_descent(f2, deriv_f2, x0, alpha, epsilon)
    plot_opt(f2, x_opt2, "function 2")

    # varied x_0
    for x0_test in [3, -3]:
        x_opt1, _ = gradient_descent(f1, deriv_f1, x0_test, alpha, epsilon)
        x_opt2, _ = gradient_descent(f2, deriv_f2, x0_test, alpha, epsilon)

    # varied alpha value
    for alpha_test in [1, 0.001, 0.0001]:
        x_opt1, iter1 = gradient_descent(f1, deriv_f1, x0, alpha_test, epsilon)

    # varied eps
    for epsilon_test in [0.1, 0.01, 0.0001]:
        x_opt1, iter1 = gradient_descent(f1, deriv_f1, x0, alpha, epsilon_test)

# %% Section 1.2

def f3(x):
    """
    Args:
        x: the input to the function

    Returns:
        function 3 applied to x
    """
    return np.sin(x) + np.cos(np.sqrt(2)*x)

def deriv_f3(x):
    """
    Args:
        x: the input to our derivative

    Returns:
        the derivative of function 3 applied to x
    """
    return np.cos(x) - np.sqrt(2)*np.sin(np.sqrt(2)*x)

def plot_f3():
    """
    Plot function 3 for 0 <= x <= 10
    """
    x = np.linspace(0, 10, 500)
    y = [f3(xi) for xi in x]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='f3(x)', line=dict(color='blue')))

    fig.update_layout(
        title='function 3',
        xaxis_title='x',
        yaxis_title='f3(x)',
        showlegend=True,
        hovermode='x'
    )

    fig.show()



def plot_opt_multiple(f, x_opts, title):
    """
    Just remaking plot_opt as another function for simplicity (this time with multiple optimal points)
    """
    x = np.linspace(0, 10, 500)
    y = [f(xi) for xi in x]

    fig = go.Figure()

    # Add function line
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Function', line=dict(color='blue')))

    # Add optimal points
    for i, x_opt in enumerate(x_opts):
        fig.add_trace(go.Scatter(x=[x_opt], y=[f(x_opt)], mode='markers',
                                name=f'Local min {i+1}: x={x_opt:.3f}',
                                marker=dict(size=10)))

    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title='f3(x)',
        showlegend=True,
        hovermode='x'
    )

    fig.show()

# %% Section 2

def approx_derivative(f, x, h=1e-5):
    """
    Args:
        f: the function we are deriving using the definition of a derivative
        x: the value of the input x we are using to estimate the derivative
        h: (default 1e-5), the small amount we move by to see the tangent
    """
    return (f(x + h) - f(x)) / h

def gradient_descent_approx(f, x0, alpha, epsilon=0.001, iter_max=1000, h=1e-5):
    """
    Writing a new gradient descent func instead of updating:
    Args:
        f: the function to descend upon
        alpha: the step size
        epsilon: the tolerance for convergence
        iter_max: (default 1000) number of iterations to run gradient descent
        h: the value we want to nudge over to see the change in tangent slope

    Returns:
        a tuple of the current value of x we are at, and the current iteration.
    """
    x = x0
    iter = 0

    while iter < iter_max:
        deriv_approx = approx_derivative(f, x, h)
        x_new = x - alpha * deriv_approx

        if abs(x_new - x) < epsilon:
            break

        x = x_new
        iter += 1

    return x, iter

# %% Section 3

def approx_partial_derivative(f, x, y, var='x', h=1e-5):
    """
    Rewriting for simplicity when running

    Args:
        f: the function we want to derive
        x: the input x value
        y: the input y value
        var: the axis we want to add our x value to
        h: the size we mini step over in the function to see our new tangent slope
    """
    if var == 'x':
        return (f(x + h, y) - f(x, y)) / h
    else:
        return (f(x, y + h) - f(x, y)) / h

def gradient_descent_2d(f, x0, y0, alpha, epsilon=0.001, iter_max=1000, h=1e-5):
    """
    Rewriting for simplicity

    Args:
        f: the function we want to descend
        x0: the starting x value
        y0: the starting y value
        alpha: the stepsize
        epsilon: the tolerance for convergence
        iter_max: (default 1000), the amount of iterations we would like to run
        """
    x, y = x0, y0
    iter = 0

    while iter < iter_max:
        df_dx = approx_partial_derivative(f, x, y, 'x', h)
        df_dy = approx_partial_derivative(f, x, y, 'y', h)

        x_new = x - alpha * df_dx
        y_new = y - alpha * df_dy

        if abs(x_new - x) < epsilon and abs(y_new - y) < epsilon:
            break

        x, y = x_new, y_new
        iter += 1

    return x, y, iter

def f_2d(x, y):
    """
    Use the provided test func f(x,y) = x^2 + y^2 to test the multi-variable gradient descent.
    """
    return x**2 + y**2

def plot_3d_surface(x_opt, y_opt):
    """
    Plot 3D surface with the optimal point in the loss landscape (using plotly)

    Args:
        x_opt: the optimal x value
        y_opt: the optimal y value

    Returns:
        nothing. Instead, it opens the chart in your browser!
    """
    x = np.linspace(-4, 4, 50)
    y = np.linspace(-4, 4, 50)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    fig = go.Figure()

    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='viridis', opacity=0.8, showscale=False))

    fig.add_trace(go.Scatter3d(x=[x_opt], y=[y_opt], z=[f_2d(x_opt, y_opt)],
                              mode='markers',
                              marker=dict(size=10, color='red'), # type: ignore
                              name='Minimum'))

    fig.update_layout(
        title='test function for 2 variables',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='f(x,y)'
        ),
        showlegend=True
    )

    fig.show()

# %% Section 4

def load_data(filename):
    """
    Load the data into two arrays

    Args:
        filename: string representing the name of the file
        containing the data (x,y)

    Return:
        array containing the values of x
        array containing the values of y
    """

    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]

def cost_function(a, b, x, y):
    """
    The given cost function for the problem to calculate our good our fit is - g(a,b).

    Args:
        a: the slope
        b: the intercept
        x: the target x
        y: the target y

    Return:
        the cost value (float)
    """

    n = len(x)
    cost = 0
    for i in range(n):
        cost += (a * x[i] + b - y[i])**2
    return cost

def plot_linear_fit(x_data, y_data, a, b, title):
    """
    Plot the line of best fit for our blood pressure data based on the cost func

    Args:
        x_data: the x values
        y_data: the y values
        a: our slope
        b: our intercept
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers',
                            name='Data points',
                            marker=dict(color='blue', size=8))) # type: ignore

    x_line = np.linspace(min(x_data), max(x_data), 100)
    y_line = a * x_line + b
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',
                            name=f'y = {a:.3f}x + {b:.3f}',
                            line=dict(color='red', width=3))) # type: ignore

    fig.update_layout(
        title=title,
        xaxis_title='Total Cholesterol Level (mmol/L)',
        yaxis_title='Diastolic Blood Pressure (mm Hg)',
        showlegend=True,
        hovermode='x'
    )

    fig.show()

def linear_regression_gradient_descent(x_data, y_data):

    """
    Find the line of best fit for the blood pressure data

    Args:
        x_data: our x values
        y_data: our y values

    Return:
        a_original: our predicted slope
        b_original: our predicted intercept
    """

    # we scale!
    x_mean, x_std = np.mean(x_data), np.std(x_data)
    y_mean, y_std = np.mean(y_data), np.std(y_data)
    x_scaled = (x_data - x_mean) / x_std
    y_scaled = (y_data - y_mean) / y_std

    def g_scaled(a, b):
        return cost_function(a, b, x_scaled, y_scaled)

    a0, b0 = 0.0, 0.0
    alpha = 0.01
    epsilon = 0.0001

    a_opt, b_opt, iterations = gradient_descent_2d(g_scaled, a0, b0, alpha, epsilon, iter_max=10000)

    # revert
    a_original = a_opt * y_std / x_std
    b_original = b_opt * y_std + y_mean - a_original * x_mean
    return a_original, b_original, iterations

# %% Section 5

def quadratic_cost_function(a, b, c, x, y):
    """
    Cost function for the given quadratic

    Args:
        a: first coefficient
        b: second coefficient
        c: c value
        x: provided x coord
        y: provided y coord

    Returns:
        the cost (float) of the current prediction.
    """

    n = len(x)
    cost = 0
    for i in range(n):
        pred = a * x[i]**2 + b * x[i] + c
        cost += (pred - y[i])**2
    return cost

def gradient_descent_3d(f, x0, y0, z0, alpha, epsilon, iter_max=10000, h=1e-5):
    """
    Perform gradient descent on a function with three variables

    Args:
        f: the function (with three variables)
        x0: initial x value
        y0: initial y value
        z0: intial z value
        alpha: stepsize
        epsilon: tolerance
        iter_max: (default 10000), max number of iterations if we dont converge before
        h: the amount we knock the point towards the bottom of the function to check the new slope

    Returns:
        x, y, z: resulting (hopefully minimum) values from grad descent
        iter: the iteration we converged at, or the max
    """
    x, y, z = x0, y0, z0
    iter = 0

    while iter < iter_max:

        #partial derivs n update
        df_dx = (f(x + h, y, z) - f(x, y, z)) / h
        df_dy = (f(x, y + h, z) - f(x, y, z)) / h
        df_dz = (f(x, y, z + h) - f(x, y, z)) / h
        x_new = x - alpha * df_dx
        y_new = y - alpha * df_dy
        z_new = z - alpha * df_dz

        if abs(x_new - x) < epsilon and abs(y_new - y) < epsilon and abs(z_new - z) < epsilon:
            break

        x, y, z = x_new, y_new, z_new
        iter += 1

    return x, y, z, iter

def plot_comparison(x_data, y_data, a_lin, b_lin, a_quad, b_quad, c_quad):
    """compare linear to quadratic"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='markers',
                            name='Data points',
                            marker=dict(color='blue', size=8))) # type: ignore

    x_line = np.linspace(min(x_data), max(x_data), 100)
    y_line_quad = a_quad * x_line**2 + b_quad * x_line + c_quad
    fig.add_trace(go.Scatter(x=x_line, y=y_line_quad, mode='lines',
                            name=f'Quadratic: y = {a_quad:.4f}x² + {b_quad:.2f}x + {c_quad:.1f}',
                            line=dict(color='green', width=3))) # type: ignore

    y_line_lin = a_lin * x_line + b_lin
    fig.add_trace(go.Scatter(x=x_line, y=y_line_lin, mode='lines',
                            name=f'Linear: y = {a_lin:.3f}x + {b_lin:.3f}',
                            line=dict(color='red', width=3, dash='dash'))) # type: ignore

    fig.update_layout(
        title='Comparison of Linear vs Quadratic function',
        xaxis_title='Total Cholesterol Level',
        yaxis_title='Diastolic Blood Pressure',
        showlegend=True,
        hovermode='x'
    )

    fig.show()


if __name__ == "__main__":
    main()

    plot_f3()

    x0_values = [1, 4, 5, 7]
    x_opts = []
    for x0 in x0_values:
        x_opt, _ = gradient_descent(f3, deriv_f3, x0, 0.1, 0.0001)
        x_opts.append(x_opt)

    plot_opt_multiple(f3, x_opts, "function 3 with local minima from different starting points")

    x_test = 2

    x_opt_approx, _ = gradient_descent_approx(f1, 3, 0.1, 0.001)
    x_opt, y_opt, iter = gradient_descent_2d(f_2d, 3, 3, 0.1, 0.001)

    plot_3d_surface(x_opt, y_opt)

    x_data, y_data = load_data('data_chol_dias_pressure.txt')

    a_opt, b_opt, iterations = linear_regression_gradient_descent(x_data, y_data)

    plot_linear_fit(x_data, y_data, a_opt, b_opt, "Regression Line plotted on Diastolic Blood Pressure vs Cholesterol")


    x_data_nl, y_data_nl = load_data('data_chol_dias_pressure_non_lin.txt')

    a_opt_lin, b_opt_lin, _ = linear_regression_gradient_descent(x_data_nl, y_data_nl)

    plot_linear_fit(x_data_nl, y_data_nl, a_opt_lin, b_opt_lin,
                    "Linear Fit on Non-linear Data")


    x_mean, x_std = np.mean(x_data_nl), np.std(x_data_nl) # type: ignore
    y_mean, y_std = np.mean(y_data_nl), np.std(y_data_nl) # type: ignore
    x_scaled = (x_data_nl - x_mean) / x_std
    y_scaled = (y_data_nl - y_mean) / y_std

    def g_quadratic(a, b, c):
        return quadratic_cost_function(a, b, c, x_scaled, y_scaled)

    a0, b0, c0 = 0.0, 0.0, 0.0
    alpha = 0.01
    epsilon = 0.0001

    a_opt_q, b_opt_q, c_opt_q, iterations = gradient_descent_3d(
        g_quadratic, a0, b0, c0, alpha, epsilon)

    a_original = a_opt_q * y_std / (x_std**2)
    b_original = b_opt_q * y_std / x_std - 2 * a_original * x_mean
    c_original = c_opt_q * y_std + y_mean - a_original * x_mean**2 - b_original * x_mean


    plot_comparison(x_data_nl, y_data_nl, a_opt_lin, b_opt_lin,
                    a_original, b_original, c_original)
