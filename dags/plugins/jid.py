from airflow.models import BaseOperator, SkipMixin
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.plugins_manager import AirflowPlugin

from airflow.utils.decorators import apply_defaults

import uuid
import getpass
import time
import requests

import logging
LOG = logging.getLogger(__name__)

class JIDSlurmOperator( BaseOperator ):
  """Airflow Operator to submit SLURM jobs through the JID.
  """

  ui_color = '#006699'

  locations = {
    'SLAC': "http://psdm02:8446/jid_slac/jid/ws/",
    'SRCF_FFB': "http://drp-srcf-mds001:8446/jid_srcf_ffb/jid/ws/",
    'NERSC': "http://psdm02:8446/jid_nersc/jid/ws/",
    'S3DF' : "https://psdm.slac.stanford.edu/arps3dfjid/jid/ws/"
  }

  btx_locations = {
    'SLAC': "/cds/sw/package/autosfx/btx/",
    'SRCF_FFB': "/cds/sw/package/autosfx/btx/",
    'NERSC': "/global/cfs/cdirs/lcls/btx/",
    'S3DF' : "/sdf/group/lcls/ds/tools/btx/"
  }
  # "/sdf/group/lcls/ds/sw/autosfx/btx/",

  endpoints = {
    'start_job': '{experiment_name}/start_job',
    'job_statuses': 'job_statuses',
    'job_log_file': '{experiment_name}/job_log_file',
  }

  @apply_defaults
  def __init__(self,
      user=getpass.getuser(),
      #run_at='SLAC',
      slurm_script=None,
      poke_interval=30,
      *args, **kwargs ):

    super(JIDSlurmOperator, self).__init__(*args, **kwargs)

    self.user = user
    #self.run_at = run_at
    #if slurm_script is None:
    #  self.slurm_script = self.get_slurm_script()
    #else:
    #  self.slurm_script = slurm_script
    self.slurm_script = slurm_script
    self.poke_interval = poke_interval

  def create_control_doc( self, context):
    """
    Create Control Doc.

    Parameters
    ----------
    context: Airflow dictionary object.
      Contains info about the current task, passed to execute().
      See: https://airflow.apache.org/docs/apache-airflow/stable/macros-ref.html#default-variables.

    Returns
    -------
    Dictionary to POST as JSON to RPC client server.

    """
    def __params_to_args__(params):
      return " ".join(["--" + k + " " + str(v) for k, v in params.items()])

    def __slurm_parameters__(params, task_id, location):
      return __params_to_args__(params) + " --task " + task_id + " --facility " + location

    return {
      "_id" : str(uuid.uuid4()),
      "arp_root_job_id": context.get('dag_run').conf.get('ARP_ROOT_JOB_ID'),
      "experiment": context.get('dag_run').conf.get('experiment'),
      "run_num" : context.get('dag_run').conf.get('run_id'),
      "user" : context.get('dag_run').conf.get('user'),
      "status" : '',
      "tool_id" : '',
      "def_id" : str(uuid.uuid4()),
      "def": {
        "_id" : str(uuid.uuid4()),
        "name" : self.task_id,
        "executable" : self.get_slurm_script(context),
        "trigger" : "MANUAL",
        "location" : self.get_location(context),
        "parameters" : __slurm_parameters__(context.get('dag_run').conf.get('parameters', {}),
                                            self.task_id, self.get_location(context)),
        "run_as_user" : self.user
      }
    }

  def get_location(self, context):
    return context.get('dag_run').conf.get("ARP_LOCATION")

  def get_slurm_script(self, context, relative_path=None):
    location = self.get_location(context)
    if not location in self.btx_locations:
      raise AirflowException(f"BTX location {location} is not configured")
    if relative_path is None:
      slurm_script = self.btx_locations[location] + "scripts/elog_submit.sh"
    else:
      slurm_script = self.btx_locations[location] + f"scripts/{relative_path}"
    #if location == 'NERSC':
    #  slurm_script = "-C cpu -A lcls " + slurm_script
    return slurm_script

  def get_file_uri( self, filepath, context ):
    location = self.get_location(context)
    if not location in self.locations:
      raise AirflowException(f"JID location {location} is not configured")
    uri = self.locations[location] + self.endpoints['file'] + filepath
    return uri

  def parse( self, resp ):
    LOG.info(f"  {resp.status_code}: {resp.content}")
    if not resp.status_code in ( 200, ):
      raise AirflowException(f"Bad response from JID {resp}: {resp.content}")
    try:
      j = resp.json()
      if not j.get('success',"") in ( True, ):
        raise AirflowException(f"Error from JID {resp}: {resp.content}")
      return j.get('value')
    except Exception as e:
      raise AirflowException(f"Response from JID not parseable: {e}")

  def put_file( self, path, content ):
    uri = self.get_file_uri( path , context )
    LOG.info( f"Calling {uri}..." )
    resp = requests.put( uri, data=content, **self.requests_opts )
    v = self.parse( resp )
    if 'status' in v and v['status'] == 'ok':
      return True
    return False

  def rpc( self, endpoint, control_doc, context, check_for_error=[] ):

    if not self.get_location(context) in self.locations:
      raise AirflowException(f"JID location {self.get_location(context)} is not configured")

    uri = self.locations[self.get_location(context)] + self.endpoints[endpoint]
    uri = uri.format(experiment_name = context.get('dag_run').conf.get('experiment'))
    LOG.info( f"Calling {uri} with {control_doc}..." )
    resp = requests.post( uri, json=control_doc, headers={'Authorization': context.get('dag_run').conf.get('Authorization')} )
    LOG.info(f" + {resp.status_code}: {resp.content.decode('utf-8')}")
    if not resp.status_code in ( 200, ):
      raise AirflowException(f"Bad response from JID {resp}: {resp.content}")
    try:
      j = resp.json()
      if not j.get('success',"") in ( True, ):
        raise AirflowException(f"Error from JID {resp}: {resp.content}")
      v = j.get('value')
      # throw error if the response has any matching text: TODO use regex
      for i in check_for_error:
        if i in v:
          raise AirflowException(f"Response failed due to string match {i} against response {v}")
      return v
    except Exception as e:
      raise AirflowException(f"Response from JID not parseable: {e}")

  def execute( self, context ):

    LOG.info(f"Attempting to run at {self.get_location(context)}...")

    # run job for it
    LOG.info("Queueing slurm job...")
    control_doc = self.create_control_doc( context )
    LOG.info(control_doc)
    LOG.info(self.locations[self.get_location(context)] + self.endpoints['start_job'])
    msg = self.rpc( 'start_job', control_doc , context)
    LOG.info(f"jobid {msg['tool_id']} successfully submitted!")
    jobs = [ msg, ]

    # FIXME: initial wait for job to queue
    time.sleep(10)
    LOG.info("Checking for job completion...")
    while jobs[0].get('status') in ('RUNNING', 'SUBMITTED'):
      jobs = self.rpc( 'job_statuses', jobs, context, check_for_error=( ' error: ', 'Traceback ' ))
      time.sleep(self.poke_interval)

    # grab logs and put into xcom
    out = self.rpc( 'job_log_file', jobs[0], context)
    context['task_instance'].xcom_push(key='log',value=out)






class JIDPlugins(AirflowPlugin):
    name = 'jid_plugins'
    operators = [JIDSlurmOperator]
