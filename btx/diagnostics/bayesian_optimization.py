"""
File: bayesian_optimization.py
Author: Paul-Emile Giacomelli
Date: Fall 2023
Description: Bayesian optimization logic to be implemented in Airflow Branch Operator and Slurm Job
"""


# Imports
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm


class BayesianOptimization:

    def __init__(self):
        pass

    # Initial samples

    def random_param_samples(config, params_ranges_keys):
        n_params = len(params_ranges_keys)
        n_samples = config.bayesian_opt.n_samples_init
        n_points = config.bayesian_opt.n_points_per_param

        random_param_samples = np.empty(shape=(n_samples, n_params))

        for i in range(n_params):
            # Get the parameter range
            param_min, param_max = config["bayesian_opt"][params_ranges_keys[i]]

            # Get n_samples random values of the parameter within its range 
            param_range = np.linspace(param_min, param_max, n_points)
            param_sample = np.random.choice(param_range, size=n_samples)

            random_param_samples[:, i] = param_sample

        # Return a matrix with each row containing a set of random parameters
        return random_param_samples
    
        
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