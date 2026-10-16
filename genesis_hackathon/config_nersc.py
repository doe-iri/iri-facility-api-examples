import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

# API access
BASE_URL = "https://api.iri.nersc.gov/api/v1"
TOKEN = os.environ.get("IRI_TOKEN_NERSC")

# NERSC account (set here or in the environment/.env; the environment takes precedence)
NERSC_USERNAME = ""
NERSC_USERNAME = os.environ.get("NERSC_USERNAME", "").strip() or NERSC_USERNAME.strip()
if NERSC_USERNAME:
    HOME_PATH = f"/global/homes/{NERSC_USERNAME[0]}/{NERSC_USERNAME}"
else:
    HOME_PATH = None

# Job submission resources
COMPUTE_RESOURCE_ID = "94351904-6dba-4c16-b5cd-fbd280d8615b" # Perlmutter
FILESYSTEM_RESOURCE_ID = "65b28619-c3b6-4942-8da1-044a3b3a2a9e" # Global home

# Job submission parameters
NODES=1
WALLTIME_SEC=300
QUEUE="regular"
COMPUTE_ALLOCATION="m5429"
# Timestamp both log files together (UTC, with microseconds).
LOG_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
STDOUT_PATH=f"{HOME_PATH}/iri_test_{LOG_TIMESTAMP}.out"
STDERR_PATH=f"{HOME_PATH}/iri_test_{LOG_TIMESTAMP}.err"

# Choose the reservation_id for your session (all times Central Daylight Time, CDT):
# gm_hackathon26_day1:     2026-10-14 10:45 to 2026-10-14 17:00
# gm_hackathon26_day2:     2026-10-15 09:00 to 2026-10-15 15:30
# gm_hackathon26_session5: 2026-10-15 15:30 to 2026-10-15 17:00
# gm_hackathon26_day3:     2026-10-16 09:00 to 2026-10-16 15:30
custom_attributes = {"reservation_id": "gm_hackathon26_day1"}
# To submit without a reservation, replace the line above with:
# custom_attributes = {}

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


