import os
import sys
from models import Config
from dotenv import load_dotenv
load_dotenv()

# API access
BASE_URL = "https://amsc-open.s3m.olcf.ornl.gov/api/v2"
TOKEN = os.environ.get("IRI_TOKEN_OLCF")
if TOKEN is None:
    print("IRI_TOKEN_OLCF missing in .env file.")
    sys.exit(1)

# OLCF project
OLCF_PROJECT = os.environ.get("OLCF_S3M_PROJECT")
if OLCF_PROJECT is None:
    print("OLCF_S3M_PROJECT missing in .env file.")
    sys.exit(1)

# Job submission resources
COMPUTE_RESOURCE_ID = "odo"
FILESYSTEM_RESOURCE_ID = "wolf2" 

# Job submission parameters
NODES=1
WALLTIME_SEC=300
QUEUE="batch"
COMPUTE_ALLOCATION=OLCF_PROJECT
WORKING_DIR=f"/gpfs/wolf2/olcf/{OLCF_PROJECT}/proj-shared"
STDOUT_PATH="iri_test.out"
STDERR_PATH="iri_test.err"

# Commands to be executed in the job
COMMANDS="""
echo Start
sleep 5
whoami
hostname
echo End
"""

# Job list
FILTERS={}

config = Config(
    base_url=BASE_URL,
    token=TOKEN,
    compute_resource_id = COMPUTE_RESOURCE_ID,
    filesystem_resource_id = FILESYSTEM_RESOURCE_ID,
    nodes = NODES,
    walltime_sec = WALLTIME_SEC,
    queue = QUEUE,
    compute_allocation = COMPUTE_ALLOCATION,
    working_directory = WORKING_DIR,
    stdout_path = STDOUT_PATH,
    stderr_path = STDERR_PATH,
    commands = COMMANDS,
    filters = FILTERS,
)
