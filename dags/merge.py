from datetime import datetime
import os
from airflow import DAG
import importlib
jid = importlib.import_module("btx-dev.plugins.jid")
JIDSlurmOperator = jid.JIDSlurmOperator

# DAG SETUP
description='BTX process SFX DAG'
dag_name = os.path.splitext(os.path.basename(__file__))[0]

dag = DAG(
    dag_name,
    start_date=datetime( 2022,4,1 ),
    schedule_interval=None,
    description=description,
  )


# Tasks SETUP
task_id='merge'
merge = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='solve'
solve = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='elog_display'
elog_display = JIDSlurmOperator(task_id=task_id, dag=dag)

# Draw the DAG
merge >> solve >> elog_display
