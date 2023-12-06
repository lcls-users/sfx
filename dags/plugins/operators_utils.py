"""
File: operators_utils.py
Author: Paul-Emile Giacomelli
Date: Fall 2023
Description: Utils functions to be called by Airflow operators
"""


# Imports


class OperatorsUtils:

    def __init__(self,
                 criterion_name=None,
                 first_loop_task=None,
                 exit_loop_task=None,
                 max_iterations=None
                 ):
        # Bayesian optimization
        if criterion_name is not None:
            if criterion_name == "max_iterations":
                    self.iteration = 0
                    self.max_iterations = max_iterations
                    self.criterion_name = criterion_name
            else:
                raise KeyError(f"The stop criterion name {criterion_name} does not exist.")
        self.first_loop_task = first_loop_task
        self.exit_loop_task = exit_loop_task

    
    def bo_stop_criterion(self):
        """
        Function called by the Airflow Branch Operator when performing a Bayesian optimization.

        Returns the name of the next task to execute given the state of the stop criterion. 

        Parameters
        ----------
        None
        """
        if self.criterion_name == "max_iterations":
            stop_criterion = self.iteration >= self.max_iterations
        else:
            pass # Already checked in class initialization
        
        if stop_criterion == False:
            # Run another iteration
            self.iteration += 1
            return self.first_loop_task + f"__bo{self.iteration:03d}"
        else:
            # Exit the loop
            return self.exit_loop_task