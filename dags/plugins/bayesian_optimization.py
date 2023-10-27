"""
File: bayesian_optimization.py
Author: Paul-Emile Giacomelli
Date: Fall 2023
Description: Bayesian optimization logic to be implemented in DAGs
"""


# Imports

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm


class BayesianOptimization:

    # Initialization

    def __init__(self):
        # TODO: get the config to perform the initializations

        # Save maximum number of iterations
        self.max_iterations = config.bayesian_opt.max_iterations
        # Initialize number of iterations
        self.iteration = 0

        # Get the initial samples
        params_ranges_keys = [key for key in config if key.startswith("range_")]
        n_params = len(params_ranges_keys)

        n_points = 100
        sample_inputs_init = np.zeros(shape=(n_points, 1))

        n_samples = config.bayesian_opt.n_samples_init

        param_ranges = []

        for i in range(n_params):
            param_min = config["bayesian_opt"][params_ranges_keys[i]][0]
            param_max = config["bayesian_opt"][params_ranges_keys[i]][1]

            param_range = np.linspace(param_min, param_max, n_points)
            param_sample_init = np.random.choice(param_range, size=n_samples)

            param_ranges.append(param_range)
            sample_inputs_init = np.column_stack((sample_inputs_init, param_sample_init))

        self.sample_inputs = sample_inputs_init

        meshes = np.meshgrid(*param_ranges, indexing='ij')
        self.input_range = np.column_stack([mesh.ravel() for mesh in meshes])

        # TODO: execute the "loop tasks chain" n_samples times, and save the scores


        # Initialize Gaussian Process
        match config.bayesian_opt.kernel:
            case "rbf":
                self.kernel = RBF(length_scale=[1.0]*n_params, length_scale_bounds=(1e-2, 1e3))
            case _:
                raise KeyError(f"The kernel {config.bayesian_opt.kernel} does not exist.")
        self.gp_model = GaussianProcessRegressor(kernel=self.kernel)

        pass

        
    # Acquisition functions

    def expected_improvement(X, gp_model, best_y):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        z = (y_pred - best_y) / y_std
        ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
        return ei

    def upper_confidence_bound(X, gp_model, beta):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        ucb = y_pred + beta * y_std
        return ucb

    def probability_of_improvement(X, gp_model, best_y):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        z = (y_pred - best_y) / y_std
        pi = norm.cdf(z)
        return pi

    def acquisition_function(self, function_name, inputs):
        output = None
        match function_name:
            case "ei" | "expected_improvement":
                output = self.expected_improvement(*inputs)
            case "ucb" | "upper_confidence_bound":
                output = self.upper_confidence_bound(*inputs)
            case "pi" | "probability_of_improvement":
                output = self.probability_of_improvement(*inputs)
            case _:
                raise KeyError(f"The acquisition function name {function_name} does not exist.")
        return output
    
    
    # Bayesian Optimization

    def run_bayesian_opt(self, loop_task_name, exit_task_name):
        if self.iteration > 0:
            # TODO: get the score from the interation that has just been run
            # new_y = black_box_function(x, new_input[0], new_input[1], new_input[2])
            # sample_y = np.append(sample_y, new_y)
            pass

        # 1. Fit the Gaussian process model to the sampled points
        self.gp_model.fit(sample_inputs, sample_y)

        # 2. Determine the point with the highest observed function value
        best_idx = np.argmax(sample_y)
        best_input = sample_inputs[best_idx]
        best_y = sample_y[best_idx]

        # 3. Generate the Acquisition Function using the Gaussian Process
        af = self.acquisition_function("expected_improvement", [self.input_range, self.gp_model, best_y])

        # 4. Select the next point based on the Acquisition Function
        if self.iteration < self.max_iterations - 1:
            new_idx = np.argmax(af)
            new_input = self.input_range[new_idx]
            sample_inputs = np.vstack((sample_inputs, new_input))

            # TODO: actually modify the parameters of the peak finding function (through config? through return?)
            
            self.iteration += 1

            # Execute the first task of the loop tasks sequence
            return loop_task_name
        else:
            # Exit the loop
            return exit_task_name