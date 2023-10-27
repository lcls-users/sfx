from datetime import datetime
import os
from airflow import DAG
from plugins.jid import JIDSlurmOperator
from airflow.operators.python import BranchPythonOperator
from plugins.bayesian_optimization import BayesianOptimization

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
task_id='find_peaks'
find_peaks = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='index'
index = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='stream_analysis'
stream_analysis = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='merge'
merge = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='solve'
solve = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='elog_display'
elog_display = JIDSlurmOperator(task_id=task_id, dag=dag)

# Instantiate a Bayesian Optimization class
bayesian_opt = BayesianOptimization()

# Branch Operator to simulate the while loop
branch = BranchPythonOperator(
    task_id='bayesian_opt_branch_task',
    python_callable=bayesian_opt.run_bayesian_opt(find_peaks.task_id, solve.task_id),
    provide_context=False,
    dag=dag
  )

# Draw the DAG
branch >> find_peaks >> index >> stream_analysis >> merge >> branch
branch >> solve >> elog_display
