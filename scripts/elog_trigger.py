#!/sdf/group/lcls/ds/ana/sw/conda1/inst/envs/ana-4.0.47-py3/bin/python

import os
import logging
import argparse
import uuid
import datetime
import getpass
#import time

import requests
from requests.auth import HTTPBasicAuth

"""
ARP initial trigger script for BTX processing.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action='store_true', help="Turn on verbose logging")
    parser.add_argument("-d", "--dag", help="Name of the DAG", default="test")
    parser.add_argument("-c", "--config", help="Absolute path to the config file ( .yaml)")
    parser.add_argument("-q", "--queue", help="The SLURM queue to be used")
    parser.add_argument("-n", "--ncores", help="Number of cores", default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    # airflow_url = "http://172.21.32.139:8080/airflow-dev/"
    airflow_s3df = "http://172.24.5.247:8080/"

    # test to make sure the Airflow API is alive and well
    resp = requests.get(airflow_s3df + "api/v1/health", auth=HTTPBasicAuth('btx', 'btx'))
    resp.raise_for_status()

    experiment_name = os.environ["EXPERIMENT"]
    run_num = os.environ["RUN_NUM"]
    auth_header = os.environ["Authorization"]

    dag_run_data = {
        "dag_run_id": str(uuid.uuid4()),
        "conf": {
            "experiment": experiment_name,
            "run_id": str(run_num) + datetime.datetime.utcnow().isoformat(),
            "JID_UPDATE_COUNTERS": os.environ["JID_UPDATE_COUNTERS"],
            "ARP_ROOT_JOB_ID": os.environ["ARP_JOB_ID"],
            "ARP_LOCATION": 'S3DF', #os.environ["ARP_LOCATION"],
            "Authorization": auth_header,
            "user": getpass.getuser(),
            "parameters": {
                "config_file": args.config,
                "dag": args.dag,
                "queue": args.queue,
                "ncores": args.ncores,
                "experiment_name": experiment_name,
                "run_number": run_num
            }
        }
    }

    resp = requests.post(airflow_s3df + f"api/v1/dags/{args.dag}/dagRuns", json=dag_run_data, auth=HTTPBasicAuth('btx', 'btx'))
    resp.raise_for_status()
    print(resp.text)

    #time.sleep(300)
