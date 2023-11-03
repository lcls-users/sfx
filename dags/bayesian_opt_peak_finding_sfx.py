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

# Tasks SETUP

# Number of initial samples (must be the same as in the config .yaml file)
n_samples_init = 5

# Generate the config files for the initial samples
task_id='bo_init_samples_configs'
bo_init_samples_configs = JIDSlurmOperator( task_id=task_id, dag=dag)

# Generate the N initial samples
for branch_id in range(1, n_samples_init + 1):

  # Define the tasks of the branch
  find_peaks = JIDSlurmOperator( task_id=f"find_peaks_{branch_id}", branch_id=branch_id, dag=dag)

  index = JIDSlurmOperator( task_id=f"index_{branch_id}", branch_id=branch_id, dag=dag)

  stream_analysis = JIDSlurmOperator( task_id=f"stream_analysis_{branch_id}", branch_id=branch_id, dag=dag)

  merge = JIDSlurmOperator( task_id=f"merge_{branch_id}", branch_id=branch_id, dag=dag)

  # Draw the branch
  bo_init_samples_configs >> find_peaks >> index >> stream_analysis >> merge
    

# Aggregate the scores and parameters of the samples
task_id='bo_aggregate_init_samples'
bo_aggregate_init_samples = JIDSlurmOperator( task_id=task_id, dag=dag)

# Define the tasks for the Bayesian optimization (post-initialization)
task_id='find_peaks'
find_peaks = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='index'
index = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='stream_analysis'
stream_analysis = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='merge'
merge = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='bayesian_optimization'
bayesian_optimization = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='solve'
solve = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='elog_display'
elog_display = JIDSlurmOperator(task_id=task_id, dag=dag)

# Branch Operator to simulate the while loop
bayesian_opt = BayesianOptimization()

branch = BranchPythonOperator(
    task_id='bayesian_opt_branch_task',
    python_callable=bayesian_opt.run_bayesian_opt(find_peaks.task_id, solve.task_id),
    provide_context=False,
    dag=dag
  )

# Draw the DAG
branch >> find_peaks >> index >> stream_analysis >> merge >> branch
branch >> solve >> elog_display
