"""
File: bayesian_optimization.py
Author: Paul-Emile Giacomelli
Date: Fall 2023
Description: Bayesian optimization logic to be implemented in Airflow Branch Operator and Slurm task
"""


# Imports
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import os
import csv
import yaml


class BayesianOptimization:

    def __init__(self, criterion_name, first_loop_task, exit_loop_task, max_iterations=None):
        match criterion_name:
            case "max_iterations":
                self.iteration = 0
                self.max_iterations = max_iterations
            case _:
                raise KeyError(f"The stop criterion name {criterion_name} does not exist.")
        self.criterion_name = criterion_name
        self.first_loop_task = first_loop_task
        self.exit_loop_task = exit_loop_task

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

    # Function to be used by Slurm task
    def run_bayesian_opt(self, config):
        setup = config.setup
        task = config.bayesian_opt
        # Get the first task of the loop sequence
        task_to_optimize = config.get(task.task_to_optimize)
        # Get the task generating the scores
        score_task = config.get(task.score_task)

        ### 1. Save the current parameters and the associated score

        # Get the current parameters
        params_ranges_keys = [key for key in config if key.startswith("range_")]
        params_names = [key.replace("range_", "") for key in params_ranges_keys]
        n_params = len(params_names)
        params = np.array([config[task_to_optimize][params_names[j]] for j in range(n_params)])

        # Get the score from the interation that has just been run
        score_file_name = f"{task.tag}_{task.fom}_n1.dat" 
        score_file_path = os.path.join("./", task.score_task, score_task.tag, "hkl", score_file_name)
        with open(score_file_path, 'r') as file:
            lines = file.readlines()
        data = lines[1].strip().split(',')
        score = data[1] # The score is located at the 2nd column

        # Save the current parameters and the associated score
        output_file_path = os.path.join(setup.root_dir, "btx", "diagnostics", f"{setup.exp}_bayesian_opt.dat")
        with open(output_file_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            # Write the data
            data_to_write = [(score, *params)]
            writer.writerows(data_to_write)
        
        ### 2. Get all samples scores and parameters

        # Read all scores and parameters from the .dat file
        with open(output_file_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)  # Skip the header row
            data_rows = list(reader)

        # Extract scores and parameters from data_rows
        sample_y = np.array([row[0] for row in data_rows], dtype=float)
        sample_inputs = np.array([row[1:] for row in data_rows], dtype=float)
        
        ### 3. Determine the next set of parameters to be used

        # 1. Fit the Gaussian process model to the sampled points
        length_scale = [1.0] * n_params
        kernel = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3))
        gp_model = GaussianProcessRegressor(kernel=kernel)
        gp_model.fit(sample_inputs, sample_y)

        # 2. Determine the point with the highest observed function value
        best_idx = np.argmax(sample_y)
        best_input = sample_inputs[best_idx]
        best_y = sample_y[best_idx]

        # 3. Generate the Acquisition Function using the Gaussian Process
        af_name = task.acquisition_function

        n_points_per_param = task.n_points_per_param
        input_range = np.zeros((n_points_per_param, n_params))
        for i, key in enumerate(params_ranges_keys):
            param_range = task.get(key)
            min_value = param_range[0]
            max_value = param_range[1]
            input_range[:, i] = np.linspace(min_value, max_value, n_points_per_param)
        
        af = self.acquisition_function(af_name, [input_range, gp_model, best_y])

        # 4. Select the next set of parameters based on the Acquisition Function
        new_idx = np.argmax(af)
        new_input = input_range[new_idx]

        # 5. Overwrite the new set of parameters in the config .yaml file
        for i, param_name in enumerate(params_names):
            task_to_optimize[param_name] = new_input[i]
        
        config_file_path = os.path.join(setup.root_dir, "btx", "tutorial", f"{setup.exp}_bayesian_opt.yaml")
        with open(config_file_path, 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

    # Function to be used by Airflow Branch Operator
    def stop_criterion(self):
        match self.criterion_name:
            case "max_iterations":
                stop_criterion = self.iteration >= self.max_iterations
            case _:
                pass # Already checked in class initialization
        
        if stop_criterion == False:
            # Run another iteration
            self.iteration += 1
            return self.first_loop_task
        else:
            # Exit the loop
            return self.exit_loop_task
    
    