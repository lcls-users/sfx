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
from plugins.jid import JIDSlurmOperator
from airflow.operators.python import BranchPythonOperator
from btx.diagnostics.bayesian_optimization import BayesianOptimization

# DAG SETUP
description='BTX Bayesian Optimization Peak Finding SFX DAG'
dag_name = os.path.splitext(os.path.basename(__file__))[0]

dag = DAG(
    dag_name,
    start_date=datetime( 2022,4,1 ),
    schedule_interval=None,
    description=description,
  )

# PARAMETERS (must be the same as in the config .yaml file)

# Number of initial samples
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

# Generate the N initial samples
for branch_id in range(1, n_samples_init + 1):

  # Define the tasks of the branch
  find_peaks = JIDSlurmOperator( task_id=f"find_peaks", branch_id=branch_id, dag=dag )

  index = JIDSlurmOperator( task_id=f"index", branch_id=branch_id, dag=dag )

  stream_analysis = JIDSlurmOperator( task_id=f"stream_analysis", branch_id=branch_id, dag=dag )

  merge = JIDSlurmOperator( task_id=f"merge", branch_id=branch_id, dag=dag )

  # Draw the branch
  bo_init_samples_configs >> find_peaks >> index >> stream_analysis >> merge >> bo_aggregate_init_samples
  

# Define the tasks for the Bayesian optimization (post-initialization)
task_id='find_peaks'
find_peaks = JIDSlurmOperator( task_id=task_id, dag=dag )

task_id='index'
index = JIDSlurmOperator( task_id=task_id, dag=dag )

task_id='stream_analysis'
stream_analysis = JIDSlurmOperator( task_id=task_id, dag=dag )

task_id='merge'
merge = JIDSlurmOperator( task_id=task_id, dag=dag )

task_id='bayesian_optimization'
bayesian_optimization = JIDSlurmOperator( task_id=task_id, dag=dag )

task_id='solve'
solve = JIDSlurmOperator( task_id=task_id, dag=dag )

task_id='elog_display'
elog_display = JIDSlurmOperator(task_id=task_id, dag=dag )

# Branch Operator to simulate the while loop
bayesian_opt = BayesianOptimization(criterion_name="max_iterations",
                                    first_loop_task="find_peaks",
                                    exit_loop_task="solve",
                                    max_iterations=max_iterations)

branch = BranchPythonOperator(
    task_id='bayesian_opt_branch_task',
    python_callable=bayesian_opt.stop_criterion(),
    provide_context=False,
    dag=dag
  )

# Draw the DAG
bo_aggregate_init_samples >> branch
branch >> find_peaks >> index >> stream_analysis >> merge >> branch
branch >> solve >> elog_display
