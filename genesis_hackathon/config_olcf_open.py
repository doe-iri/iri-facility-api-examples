import os
from dotenv import load_dotenv
load_dotenv()

# API access
BASE_URL = "https://amsc-open.s3m.olcf.ornl.gov/api/v2"
TOKEN = os.environ.get("IRI_TOKEN_OLCF")

# OLCF project
OLCF_PROJECT = os.environ.get("OLCF_S3M_PROJECT")

# Job submission resources
COMPUTE_RESOURCE_ID = "odo"
FILESYSTEM_RESOURCE_ID = "wolf2" 

# Job submission parameters
NODES=1
WALLTIME_SEC=300
QUEUE="batch"
COMPUTE_ALLOCATION=OLCF_PROJECT
WORKING_DIR=f"/gpfs/wolf2/olcf/{OLCF_PROJECT}/proj-shared" if OLCF_PROJECT else None
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


