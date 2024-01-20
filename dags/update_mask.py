from datetime import datetime
import os
from airflow import DAG
import importlib
jid = importlib.import_module("btx-dev.plugins.jid")
JIDSlurmOperator = jid.JIDSlurmOperator

# DAG SETUP
description='BTX update mask DAG'
dag_name = os.path.splitext(os.path.basename(__file__))[0]

dag = DAG(
    dag_name,
    start_date=datetime( 2022,4,1 ),
    schedule_interval=None,
    description=description,
  )


# Tasks SETUP

task_id='build_mask'
build_mask = JIDSlurmOperator( task_id=task_id, dag=dag)

# Draw the DAG
build_mask
