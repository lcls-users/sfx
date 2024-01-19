"""
File: bayesian_opt_peak_finding_sfx.py
Author: Paul-Emile Giacomelli
Date: Fall 2023
Description: DAG for Bayesian optimization applied to peak finding
"""


# Imports
from datetime import datetime
import os
from airflow import DAG
from airflow.operators.python import BranchPythonOperator
import importlib
jid = importlib.import_module("btx-dev.plugins.jid")
JIDSlurmOperator = jid.JIDSlurmOperator
operators_utils = importlib.import_module("btx-dev.plugins.operators_utils")
OperatorsUtils = operators_utils.OperatorsUtils

# DAG SETUP
description='BTX Bayesian Optimization Peak Finding SFX DAG'
dag_name = os.path.splitext(os.path.basename(__file__))[0]

dag = DAG(
    dag_name,
    start_date=datetime( 2022,4,1 ),
    schedule_interval=None,
    description=description,
  )

# PARAMETERS

# Number of initial samples (must be the same as in the config .yaml file)
n_samples_init = 5
# Maximum number of iterations
max_iterations = 10

# Tasks SETUP

# Generate the config files for the initial samples
task_id='bo_init_samples_configs'
bo_init_samples_configs = JIDSlurmOperator( task_id=task_id, dag=dag )

# Task to aggregate the scores and parameters of the samples
task_id='bo_aggregate_init_samples'
bo_aggregate_init_samples = JIDSlurmOperator( task_id=task_id, dag=dag )

# Generate the initial samples
for branch_id in range(1, n_samples_init + 1):

  # Define the tasks of the branch
  find_peaks = JIDSlurmOperator( task_id=f"find_peaks__sample{branch_id:03d}", branch_id=branch_id, dag=dag)

  index = JIDSlurmOperator( task_id=f"index__sample{branch_id:03d}", branch_id=branch_id, dag=dag )

  stream_analysis = JIDSlurmOperator( task_id=f"stream_analysis__sample{branch_id:03d}", branch_id=branch_id, dag=dag )

  merge = JIDSlurmOperator( task_id=f"merge__sample{branch_id:03d}", branch_id=branch_id, dag=dag )

  # Draw the branch
  bo_init_samples_configs >> find_peaks >> index >> stream_analysis >> merge >> bo_aggregate_init_samples
  


# Branch Operator
op_utils = OperatorsUtils(criterion_name="max_iterations",
                                    first_loop_task="find_peaks",
                                    exit_loop_task="solve",
                                    max_iterations=max_iterations)

# Generate the branch operators
branch_operators = []

for id in range(1, max_iterations + 2):
  branch = BranchPythonOperator(
    task_id=f'bayesian_opt_branch_task_{id}',
    python_callable=op_utils.bo_stop_criterion,
    dag=dag
  )
  branch_operators.append(branch)

# Connect the aggregate task to the first branch operator
bo_aggregate_init_samples >> branch_operators[0]

# Draw all branches
for i in range(max_iterations):
  # Define the tasks for this iteration
  find_peaks = JIDSlurmOperator( task_id=f'find_peaks__bo{i+1:03d}', dag=dag )

  index = JIDSlurmOperator( task_id=f'index__bo{i+1:03d}', dag=dag )

  stream_analysis = JIDSlurmOperator( task_id=f'stream_analysis__bo{i+1:03d}', dag=dag )

  merge = JIDSlurmOperator( task_id=f'merge__bo{i+1:03d}', dag=dag )

  bayesian_optimization = JIDSlurmOperator( task_id=f'bayesian_optimization__bo{i+1:03d}', dag=dag )

  # Draw the branch
  branch_operators[i] >> find_peaks >> index >> stream_analysis >> merge >> bayesian_optimization >> branch_operators[i+1]


# Exit
solve = JIDSlurmOperator( task_id='solve', dag=dag )

elog_display = JIDSlurmOperator(task_id='elog_display', dag=dag )

branch_operators[-1] >> solve >> elog_display


