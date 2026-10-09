import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# API access
# ESnet accepts AmSC tokens (PATs) on its v2 API only; v1 does not.
BASE_URL = "https://iri-dev.ppg.es.net/api/v2"
TOKEN = os.environ.get("AMSC_TOKEN")
if TOKEN is None:
    token_file = Path(os.environ.get("AMSC_TOKEN_FILE", "/tmp/amsc-token.txt"))
    if token_file.is_file():
        TOKEN = token_file.read_text().strip() or None

# Job submission resources
COMPUTE_RESOURCE_ID = "fb0aafe1-c780-55c0-b635-a7121f1b0ce5" # Low Priority compute
FILESYSTEM_RESOURCE_ID = "b6d59ba3-5c2d-57dd-b3b9-108832da0578" # Storage

# Job submission parameters
# ESnet schedules jobs with Kueue: QUEUE and COMPUTE_ALLOCATION are accepted but not used.
NODES=1
WALLTIME_SEC=300
QUEUE="default"
COMPUTE_ALLOCATION="hackathon2610-project"
# Unique log names per submission: all hackathon users share this home directory.
LOG_ID = uuid.uuid4()
STDOUT_PATH=f"/data/home/hackathon2610/iri_test_{LOG_ID}.out"
STDERR_PATH=f"/data/home/hackathon2610/iri_test_{LOG_ID}.err"

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


