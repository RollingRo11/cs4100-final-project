# how to use + Report

## How to use

### For pip:
- `pip install numpy plotly`
- `python`/`python3` project.py
- all of the charts should open up in your browser (interactively!)

### for uv:
- `uv install` or `uv add numpy plotly`
- `uv run project.py`
- all of the charts should open up in your browser (interactively!)


## Report:

### Initial design derivations:
- I'm currently writing this on the plane haha! I initially planned to attempt to write a crosscoder as an experiment, but I got on the plane and realized I didn't have service (but luckily, I had the default class project downloaded and some plotly docs saved on my computer). I'll probably have my first attempt at a crosscoder done by Wednesday, June 25th (I'll send you the repo along with some papers we can try to read by Friday tomorrow).
- I used plotly (it's used a lot in mechanistic interpretability and I thought it would be good practice to get used to it here)
- I opted to rewrite functions rather than modify so I could just run the entire file without having to hardcode None cases. I think I still was able to achieve what the project was getting at, since I did iterate on the functions I rewrote.

> [!NOTE]
> _"Write a report that explains how you designed your code to implement gradient descent for one variable first and how you generalized it to several variables. Explain how you tested your code with different cases. Summarize what you learnt when applying a perturbation analysis of the parameters, also known as parameter hyper tuning, and how different values of x0, α and ϵcan affect the performance of gradient descent. Explain how you designed your code to implement gradient descent for functions of several variables on the cost function g(a,b). Summarize what you learnt about how the choice of f(x) can affect the performance on the model depending on the data set."_

### The gradient descent function:

I opted to rewrite functions rather than physically changing them, but the idea of adapting them is the same. First, I implemented the standard gradient descent algorithm. I allowed the user to provide the function, the derivative of the function, the alpha value, the tolerance (which I defaulted to 1e-5, which might've been a little too small, but the code I ended up writing overwrote this anyway), and a max iterations of 1000.

I then adapted this function by writing `gradient_descent_approx`, which instead of being provided the derivative of the function, automatically estimates the derivative by using the definition of the derivative and "nudging" the value over by a small h value (1e-5) to see how the slope progresses between the point and the nudged point.

I then further adapted this function by writing `gradient_descent_2d`, which calculates the gradients across 2 dimensions. To do this, I calculated the partial derivative with respect to each of the two variables and descended upon each one until they lied in the tolerance range.

I then wrote `linear_regression_gradient_descent`, which calculated the line of best fit for our blood pressure data by taking in the `x_data`/`y_data`. I then normalized/scaled the data so that the descent would be cleaner, and set the initial a and b values to 0.0. To return the values, I "unscaled" them and returned them!


### Parameter Hypertuning:

I think the biggest thing I realized was how sensitive it hyperparameter/parameter tuning would be even for smaller models. I have primarily experimented with Large Language Models, and they're pretty resistant to an off hyperparameter but they still get affected quite a lot. I guess it should've been obvious that the smaller and less complex the model, the more sensitive it would be to hyperparameter tuning.

### Choice of function:

The choice of function highly depends on the pattern of the plotted data. It's self explanatory that curved data may require some exponential or quadratic, but the difference in performance was still striking and I think it was cool to create the visualization that compared the quadratic to the linear function. It really showed how data-dependent the choice of model is.
