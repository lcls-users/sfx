from datetime import datetime
import os
from airflow import DAG
import importlib
jid = importlib.import_module("btx-dev.plugins.jid")
JIDSlurmOperator = jid.JIDSlurmOperator

# DAG SETUP
description='BTX build pump probe mask DAG'
dag_name = os.path.splitext(os.path.basename(__file__))[0]

dag = DAG(
    dag_name,
    start_date=datetime( 2022,4,1 ),
    schedule_interval=None,
    description=description,
  )


# Tasks SETUP
task_id='make_histogram'
find_peaks = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='measure_emd'
elog1 = JIDSlurmOperator(task_id=task_id, dag=dag)

task_id='build_pump_probe_mask'
index = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='elog_display'
elog_display = JIDSlurmOperator(task_id=task_id, dag=dag)

# Draw the DAG
make_histogram >> measure_emd >> build_pump_probe_mask >> elog_display
