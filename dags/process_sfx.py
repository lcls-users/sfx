from datetime import datetime
import os
from airflow import DAG
from plugins.jid import JIDSlurmOperator

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

# Draw the DAG
find_peaks >> index >> stream_analysis >> merge >> solve >> elog_display
