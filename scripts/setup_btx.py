"""Prepare eLog jobs for btx workflows.

Functions
---------
determine_configuration(args, workflow)
    Parse provided arguments and return a formatted command for job submission.
"""
import os
import argparse
import requests
import json
import logging
import re
from typing import List, Dict, Tuple, Any, Optional

from krtc import KerberosTicket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def determine_configuration(
        args: argparse.Namespace,
        workflow: Optional[str] = None
) -> Optional[Tuple[str, str, str]]:
    """Parse arguments and format btx job submission command.

    @param args (argparse.Namespace) Arguments parsed by argparse.Parser.
    @param workflow (str | None) If provided, override the workflow provided
        by a command-line argument.
    @returns executable,param_string (str, str) The btx executable and the
        formatted parameter string.
    """
    if not args.experiment:
        logger.info("No experiment provided and cannot find one! ABORTING!")
        return

    exp: str = args.experiment
    dag: str = ""
    if workflow:
        dag = workflow
    else:
        if args.workflow == "sfx":
            dag = "process_sfx"
        elif args.workflow == "geometry" or args.workflow == "behenate":
            dag = "optimize_geometry"

    btx_base_dir: str = "/sdf/group/lcls/ds/tools/btx/dev/scripts"
    btx_executable: str = "elog_trigger.py"
    executable: str = f"{btx_base_dir}/{btx_executable}"

    exp_dir: str = f"/sdf/data/lcls/ds/{exp[:3]}/{exp}"
    config_file: str = f"{exp_dir}/scratch/btx/yamls/config.yaml"
    param_string: str = (
        f"-a {args.account} -c {config_file} -d {dag} -n {args.ncores}"
        f" -q {args.queue}"
    )

    reservation: str = args.reservation
    if reservation:
        param_string += f" -r {reservation}"

    if args.verbose:
        param_string += f" --verbose"
    return exp, executable, param_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--account",
        type=str,
        help="Account to use for batch jobs. Defaults to lcls:$EXP",
        default=f"lcls:{os.environ.get('EXPERIMENT', '')}"
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="Experiment to perform btx setup for.",
        default=os.environ.get("EXPERIMENT", "")
    )
    parser.add_argument(
        "-n",
        "--ncores",
        type=int,
        help="Number of cores to use for jobs. Defaults to 64.",
        default=64
    )
    parser.add_argument(
        "-q",
        "--queue",
        type=str,
        help="Queue to run on. Defaults to milano.",
        default="milano"
    )
    parser.add_argument(
        "-r",
        "--reservation",
        type=str,
        help="Reservation if there is one. Defaults to None.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Turn on verbose logging."
    )
    parser.add_argument(
        "-w",
        "--workflow",
        type=str,
        help=(
            "Which analysis workflow to run. Defaults to sfx. Options: sfx, "
            "geometry"
        ),
        default="sfx"
    )
    args = parser.parse_args()
    res = determine_configuration(args)
    workflows: List[Dict[str, str]] = []
    if res:
        exp, executable, params = res
        main_workflow: Dict[str, str] = {
            "name": "run_btx",
            "executable": executable,
            "trigger": "END_OF_RUN",
            "location": "S3DF",
            "parameters": params
        }
        workflows.append(main_workflow)
        if "optimize_geometry" not in main_workflow["parameters"]:
            geom_params: str = re.sub(
                "-d.*-n",
                "-d optimize_geometry -n",
                params
            )
            geometry_workflow: Dict[str, str] = {
                "name": "optimize_geometry",
                "executable": executable,
                "trigger": "MANUAL",
                "location": "S3DF",
                "parameters": geom_params
            }
            workflows.append(geometry_workflow)

        for workflow in workflows:
            krbticket: Any = KerberosTicket("HTTP@pswww.slac.stanford.edu")
            krbheaders: dict = krbticket.getAuthHeaders()
            url: str = (
                f"https://pswww.slac.stanford.edu/ws-kerb/lgbk/lgbk/{exp}/ws"
                "/create_update_workflow_def"
            )
            post_params: Dict[str, Any] = {
                "url": url,
                "headers": krbheaders,
                "json": workflow,
            }
            resp: requests.models.Response = requests.post(**post_params)
            resp.raise_for_status()
            # Extra logging and such...
