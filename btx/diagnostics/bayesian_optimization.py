"""
File: bayesian_optimization.py
Author: Paul-Emile Giacomelli
Date: Fall 2023
Description: Bayesian optimization logic to be implemented in Airflow Branch Operator and Slurm task
"""


# Imports
import math
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm
import os
import csv
import yaml
import copy
import shutil


class BayesianOptimization:

    def __init__(self, criterion_name, first_loop_task, exit_loop_task, max_iterations=None):
        if criterion_name == "max_iterations":
                self.iteration = 0
                self.max_iterations = max_iterations
        else:
            raise KeyError(f"The stop criterion name {criterion_name} does not exist.")
        self.criterion_name = criterion_name
        self.first_loop_task = first_loop_task
        self.exit_loop_task = exit_loop_task

    # Initial samples

    @staticmethod
    def random_param_samples(task, n_params, params_ranges_keys, n_samples, n_points):
        """
        Function called by the "bo_init_samples_configs" task.

        Creates a matrix with each row containing a set of random parameters.
        Each parameter value falls into the range that has been defined in the config.

        Parameters
        ----------
        task : AttrDict
            The "bayesian_optimization" section of "config".
        n_params: Int
            The number of parameters.
        params_ranges_keys: List
            The "range_X" keys (with X being a parameter name) to access the ranges of values for each parameter.
        n_samples: Int
            The number of initial samples for the Bayesian optimization.
        n_points: Int
            The number of points to use in linspaces.
        """
        random_param_samples = np.empty(shape=(n_samples, n_params))

        for i in range(n_params):
            # Get the parameter range
            param_min, param_max = task[params_ranges_keys[i]]

            # Get n_samples random values of the parameter within its range 
            param_range = np.linspace(param_min, param_max, n_points)
            param_sample = np.random.choice(param_range, size=n_samples)

            random_param_samples[:, i] = param_sample

        # Return a matrix with each row containing a set of random parameters
        return random_param_samples
    
        
    # Acquisition functions

    @staticmethod
    def expected_improvement(X, gp_model, best_y):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        z = (y_pred - best_y) / y_std
        ei = (y_pred - best_y) * norm.cdf(z) + y_std * norm.pdf(z)
        return ei

    @staticmethod
    def upper_confidence_bound(X, gp_model, beta):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        ucb = y_pred + beta * y_std
        return ucb

    @staticmethod
    def probability_of_improvement(X, gp_model, best_y):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        z = (y_pred - best_y) / y_std
        pi = norm.cdf(z)
        return pi
    
    
    # Bayesian Optimization

    @classmethod
    def run_bayesian_opt(cls, config):
        """
        Function called by the "bayesian_optimization" task.

        Runs one iteration of the Bayesian optimization.

        Parameters
        ----------
        config : AttrDict
            The current config.
        """
        setup = config.setup
        task = config.bayesian_optimization
        # Get the first task of the loop sequence
        task_to_optimize = config.get(task.task_to_optimize)
        # Get the task generating the scores
        score_task = config.get(task.score_task)

        ##### 1. Save the current parameters and the associated score

        # Get the current parameters
        n_params, params, params_names, params_ranges_keys= cls.get_parameters(task, task_to_optimize)
        print(params)

        # Get the score from the interation that has just been run
        score = cls.get_last_score(setup, task, score_task)
        print(score)

        # Save the current parameters and the associated score
        cls.save_iteration(setup, task, score, params)
        
        ##### 2. Get all samples scores and parameters
        sample_y, sample_inputs = cls.read_bo_history(setup, task)
        
        ##### 3. Determine the next set of parameters to be used

        # 1. Fit the Gaussian process model to the sampled points
        length_scale = [1.0] * n_params
        if task.kernel == "rbf":
            kernel = RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3))
        else:
            raise KeyError(f"The kernel {task.kernel} does not exist.")
        gp_model = GaussianProcessRegressor(kernel=kernel)
        gp_model.fit(sample_inputs, sample_y)

        # 2. Determine the point with the best observed function value
        if task.opt_type == "max":
            best_idx = np.argmax(sample_y)
        elif task.opt_type == "min":
            best_idx = np.argmin(sample_y)
        else:
            raise KeyError(f"The optimization type {task.opt_type} does not exist.")
        best_input = sample_inputs[best_idx]

        # 3. Generate the Acquisition Function using the Gaussian Process
        af_name = task.acquisition_function
        if hasattr(cls, af_name):
            af = getattr(cls, af_name)
        else:
            raise KeyError(f"The acquisition function name {af_name} does not exist.")

        n_points_per_param = task.n_points_per_param
        input_range = np.zeros((n_points_per_param, n_params))
        for i, key in enumerate(params_ranges_keys):
            param_range = task.get(key)
            min_value = param_range[0]
            max_value = param_range[1]
            input_range[:, i] = np.linspace(min_value, max_value, n_points_per_param)

        if af_name == "upper_confidence_bound":
            beta = task.beta
            inputs = [input_range, gp_model, beta]
        else:
            best_y = sample_y[best_idx]
            inputs = [input_range, gp_model, best_y]

        af_values = af(*inputs)
        print(af_values)
        
        # 4. Select the next set of parameters based on the Acquisition Function
        new_idx = np.argmax(af_values)
        print(new_idx)
        new_input = input_range[new_idx]
        print(new_input)

        # 5. Overwrite the new set of parameters in the config .yaml file
        cls.overwrite_params(config, setup, task, params_names, new_input)

    @classmethod
    def init_samples_configs(cls, config, logger):
        """
        Function called by the "bo_init_samples_configs" task.

        Generates the config files for each initial sample.

        Parameters
        ----------
        config : AttrDict
            The current config.
        logger: Logger
            The logger used to report the progess of the tasks.
        """
        setup = config.setup
        task = config.bayesian_optimization
        taskdir = os.path.join(setup.root_dir, "bayesian_opt")
        os.makedirs(taskdir, exist_ok=True)
        os.makedirs(os.path.join(taskdir, task.tag), exist_ok=True)

        if hasattr(task, 'n_samples_init'):
            # Get the first task of the loop sequence
            task_to_optimize = config.get(task.task_to_optimize)

            # Get the parameters to overwrite
            n_params, _, params_names, params_ranges_keys= cls.get_parameters(task, task_to_optimize)

            # Generate the parameter samples
            n_samples = task.n_samples_init
            n_points = task.n_points_per_param
            param_samples = cls.random_param_samples(task, n_params, params_ranges_keys, n_samples, n_points)

            # Get the task generating the scores
            score_task = config.get(task.score_task)

            # Generate the subdir
            subdir_name = setup.exp + "_init_samples_configs"
            subdir_path = os.path.join(setup.root_dir, "bayesian_opt", subdir_name)
            os.makedirs(subdir_path, exist_ok=True)

            # Generate the config files
            logger.info(f'Generating {n_samples} config files.')
            for i in range(n_samples):
                config_temp = copy.copy(config)
                # Overwrite the parameters
                for j in range(n_params):
                    config_temp[task.task_to_optimize][params_names[j]] = float(param_samples[i,j])

                # Overwrite the tags to change the name of the written files to avoid conflicts
                if hasattr(task, 'loop_tasks'):
                    loop_tasks = task.loop_tasks
                else:
                    # Default loop tasks
                    loop_tasks = ["find_peaks", "index", "stream_analysis", "merge"]

                for task_name in loop_tasks:
                    if task_name == "index":
                        config_temp[task_name]["tag_cxi"] = config_temp[task_name]["tag_cxi"] + f"_sample_{i+1}"
                    config_temp[task_name]["tag"] = config_temp[task_name]["tag"] + f"_sample_{i+1}"


                # Write the config file
                config_file_name = f"{setup.exp}_sample_{i+1}.yaml"
                config_file_path = os.path.join(subdir_path, config_file_name)

                config_temp_dict = {key: dict(value) if value is not None else None for key, value in config_temp.items()}

                with open(config_file_path, 'w') as file:
                    yaml.dump(config_temp_dict, file, default_style=False)

            # Generate the cell files
            logger.info(f'Generating {n_samples} cell files.')
            cell_dir_path = os.path.join(setup.root_dir, "cell")
            for i in range(n_samples):
                shutil.copy(os.path.join(cell_dir_path, f"{task.tag}.cell"), os.path.join(cell_dir_path, f"{task.tag}_sample_{i+1}.cell"))
                
            logger.info('Done!')
        else:
            raise NameError('The number of config files to generate was not defined!')
    
    @classmethod
    def aggregate_init_samples(cls, config, logger):
        """
        Function called by the "bo_aggregate_init_samples" task.

        Saves all scores and parameters of the initial samples into a .dat file.

        Parameters
        ----------
        config : AttrDict
            The current config.
        logger: Logger
            The logger used to report the progess of the tasks.
        """
        setup = config.setup
        task = config.bayesian_optimization
        n_samples = task.n_samples_init
        # Get the first task of the loop sequence
        task_to_optimize = config.get(task.task_to_optimize)
        # Get the task generating the scores
        score_task = config.get(task.score_task)
        # Get the names of the parameters
        n_params, _, params_names, _ = cls.get_parameters(task, task_to_optimize)
        # Get the score and the parameters of each sample
        samples_scores = np.empty(shape=(n_samples,))
        samples_params = np.empty(shape=(n_samples, n_params))

        logger.info('Aggregating the scores and parameters of the initial samples.')
        subdir_name = setup.exp + "_init_samples_configs"
        subdir_path = os.path.join(setup.root_dir, "bayesian_opt", subdir_name)
        for i in range(n_samples):
            # Get the parameters in the config file
            config_file_name = f"{setup.exp}_sample_{i+1}.yaml"
            config_file_path = os.path.join(subdir_path, config_file_name)
            with open(config_file_path, 'r') as file:
                config_data = yaml.safe_load(file)
            for j in range(n_params):
                samples_params[i,j] = config_data[task.task_to_optimize][params_names[j]]

            # Get the score in the score output file
            score_file_name = f"{task.tag}_sample_{i+1}_{task.fom}_n1.dat" 
            score_file_path = os.path.join(setup.root_dir, task.score_task, f"{score_task.tag}_sample_{i+1}", "hkl", score_file_name)
            with open(score_file_path, 'r') as file:
                lines = file.readlines()
            data = lines[1].strip().split(',')
            elements = data[0].split() # Split the string into a list of elements
            score = elements[1] # The score is located at the 2nd column
            samples_scores[i] = score

        # Write the scores and the parameters of the samples in a .dat file
        output_file_path = os.path.join(setup.root_dir, "bayesian_opt", task.tag, f"{setup.exp}_bayesian_opt.dat")
        with open(output_file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            # Write a header row with column names
            writer.writerow(['score'] + params_names)
            # Write the data
            data_to_write = [(score, *params) for score, params in zip(samples_scores, samples_params)]
            writer.writerows(data_to_write)

        # Delete the unnecessary cell files
        logger.info(f'Deleting {n_samples} cell files.')
        cell_dir_path = os.path.join(setup.root_dir, "cell")
        for i in range(n_samples):
            os.remove(os.path.join(cell_dir_path, f"{task.tag}_sample_{i+1}.cell"))


        logger.info('Done!')


    # Utils

    @staticmethod
    def get_parameters(task, task_to_optimize):
        """
        Get the parameters number, values, names, and the keys to access their ranges.

        Parameters
        ----------
        task : AttrDict
            The "bayesian_optimization" section of "config".
        task_to_optimize : AttrDict
            The section of "config" corresponding to the task to optimize.
        """
        params_ranges_keys = [key for key in task if key.startswith("range_")]
        params_names = [key.replace("range_", "") for key in params_ranges_keys]
        n_params = len(params_names)
        params = np.array([task_to_optimize[params_names[j]] for j in range(n_params)])
        return n_params, params, params_names, params_ranges_keys

    @staticmethod
    def get_last_score(setup, task, score_task):
        """
        Get the score of the latest iteration of the Bayesian optimization.

        Parameters
        ----------
        setup : AttrDict
            The "setup" section of "config".
        task : AttrDict
            The "bayesian_optimization" section of "config".
        score_task : AttrDict
            The section of "config" corresponding to the task generating the scores.
        """
        score_file_name = f"{task.tag}_{task.fom}_n1.dat" 
        score_file_path = os.path.join(setup.root_dir, task.score_task, score_task.tag, "hkl", score_file_name)
        with open(score_file_path, 'r') as file:
            lines = file.readlines()
        data = lines[1].strip().split(',')
        elements = data[0].split() # Split the string into a list of elements
        score = elements[1] # The score is located at the 2nd column
        return score
    
    @staticmethod
    def save_iteration(setup, task, score, params):
        """
        Saves an iteration of the Bayesian optimization (score and parameters) in a .dat file.

        Parameters
        ----------
        setup : AttrDict
            The "setup" section of "config".
        task : AttrDict
            The "bayesian_optimization" section of "config".
        score : float
            The score of the iteration.
        params : List
            The values of the parameters of the iteration.
        """
        output_file_path = os.path.join(setup.root_dir, "bayesian_opt", task.tag, f"{setup.exp}_bayesian_opt.dat")
        with open(output_file_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            # Write the data
            data_to_write = [(score, *params)]
            writer.writerows(data_to_write)

    @staticmethod
    def read_bo_history(setup, task):
        """
        Read the score and the parameters of all past iterations from a .dat file.

        Parameters
        ----------
        setup : AttrDict
            The "setup" section of "config".
        task : AttrDict
            The "bayesian_optimization" section of "config".
        """
        output_file_path = os.path.join(setup.root_dir, "bayesian_opt", task.tag, f"{setup.exp}_bayesian_opt.dat")
        # Read all scores and parameters from the .dat file
        with open(output_file_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            next(reader)  # Skip the header row
            data_rows = list(reader)

        # Extract scores and parameters from data_rows
        sample_y = np.array([row[0] for row in data_rows], dtype=float)
        sample_inputs = np.array([row[1:] for row in data_rows], dtype=float)

        return sample_y, sample_inputs
    
    @staticmethod
    def overwrite_params(config, setup, task, params_names, new_input):
        """
        Overwrite the new parameters in the config .yaml file.

        Parameters
        ----------
        config : AttrDict
            The current config.
        setup : AttrDict
            The "setup" section of "config".
        task : AttrDict
            The "bayesian_optimization" section of "config".
        params_names: List
            The names of the parameters to overwrite.
        new_input:
            The new values of the parameters.
        """    
        config_file_path = os.path.join(setup.root_dir, "yamls", f"{setup.exp}_bayesian_opt.yaml")
        config_dict = {key: dict(value) if value is not None else None for key, value in config.items()}
        # Overwrite the parameters
        for i, param_name in enumerate(params_names):
            # config_dict[task.task_to_optimize][param_name] = float(new_input[i])
            config_dict[task.task_to_optimize][param_name] = 50
        # Overwrite the config file
        with open(config_file_path, 'w') as yaml_file:
            yaml.dump(config_dict, yaml_file, default_flow_style=False)
    
        
    
    