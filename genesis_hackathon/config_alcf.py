"""

TODO: Please insert your ALCF username in `alcf_username`.
TODO: If you target the Flare or Eagle filesystems, please insert your ALCF project in `alcf_project`.

By default, the configuration will target Polaris and the Home Filesystem.
Below are resource IDs you can use if you want to change the configuration.

Compute Resource IDs
--------------------
COMPUTE_RESOURCE_ID = "0325fc07-6fb7-4453-b772-3d5030b2df72" # Aurora
COMPUTE_RESOURCE_ID = "55c1c993-1124-47f9-b823-514ba3849a9a" # Polaris
COMPUTE_RESOURCE_ID = "8b9b42f7-572a-4909-8472-a0453436304c" # Crux

Filesystem Resource IDs
-----------------------
FILESYSTEM_RESOURCE_ID = "154bb3be-5d12-4a76-a16b-898b8e310a4b" # Flare
FILESYSTEM_RESOURCE_ID = "1c3ad9d4-2e91-42bc-becb-72b1fde1235c" # Eagle
FILESYSTEM_RESOURCE_ID = "6115bd2c-957a-4543-abff-5fae52992ff2" # Home (for Polaris and Crux only)

"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

# TODO: Please insert your ALCF username in `alcf_username`.
# TODO: If you target the Flare or Eagle filesystems, please insert your ALCF project in `alcf_project`.
alcf_username = ""
alcf_project = "GenesisHackathonOct26"

# Target resources
COMPUTE_RESOURCE_ID = "55c1c993-1124-47f9-b823-514ba3849a9a" # Polaris
FILESYSTEM_RESOURCE_ID = "6115bd2c-957a-4543-abff-5fae52992ff2" # Home (for Polaris and Crux only)

# Job submission parameters
NODES = 1
WALLTIME_SEC = 300
QUEUE = "debug"
COMPUTE_ALLOCATION = "GenesisHackathonOct26"

# Commands to be executed in the job
COMMANDS="""
echo Start
sleep 5
whoami
hostname
echo End
"""

# Optinoal filters when listing jobs
#FILTERS={"accountingId": "alcf_training", "states": ["completed"]}
FILTERS={}


# -------------------------------------------------------------
# --------- From this point, there is nothing to edit ---------
# -------------------------------------------------------------


# API access
BASE_URL = "https://api.alcf.anl.gov/api/v1"
TOKEN = os.environ.get("IRI_TOKEN_ALCF")

# Set stdout/sdterr paths
if FILESYSTEM_RESOURCE_ID == "6115bd2c-957a-4543-abff-5fae52992ff2":
    STDOUT_PATH = f"/home/{alcf_username}/iri_test.out"
    STDERR_PATH = f"/home/{alcf_username}/iri_test.err"
elif FILESYSTEM_RESOURCE_ID == "1c3ad9d4-2e91-42bc-becb-72b1fde1235c":
    STDOUT_PATH = f"/eagle/{alcf_project}/{alcf_username}_iri_test.out"
    STDERR_PATH = f"/eagle/{alcf_project}/{alcf_username}_iri_test.err"
elif FILESYSTEM_RESOURCE_ID == "154bb3be-5d12-4a76-a16b-898b8e310a4b":
    STDOUT_PATH = f"/flare/{alcf_project}/{alcf_username}_iri_test.out"
    STDERR_PATH = f"/flare/{alcf_project}/{alcf_username}_iri_test.err"
else:
    STDOUT_PATH = None
    STDERR_PATH = None

# Automatic assignment of custom attributes
if COMPUTE_RESOURCE_ID in ["55c1c993-1124-47f9-b823-514ba3849a9a", "8b9b42f7-572a-4909-8472-a0453436304c"]:
    custom_attributes = {"filesystems": "home:eagle"}
elif COMPUTE_RESOURCE_ID == "0325fc07-6fb7-4453-b772-3d5030b2df72":
    custom_attributes = {"filesystems": "flare"}
else:
    custom_attributes = {}

# Function to validate user selection
def validate_input():

    # Compute resource selection
    if COMPUTE_RESOURCE_ID not in [
        "0325fc07-6fb7-4453-b772-3d5030b2df72",
        "55c1c993-1124-47f9-b823-514ba3849a9a",
        "8b9b42f7-572a-4909-8472-a0453436304c",
    ]:
        print(f"Compute ID {COMPUTE_RESOURCE_ID} not supported. Please adjust config_alcf.py.")
        sys.exit(1)

    # Filesystem resource selection
    if FILESYSTEM_RESOURCE_ID not in [
        "154bb3be-5d12-4a76-a16b-898b8e310a4b",
        "1c3ad9d4-2e91-42bc-becb-72b1fde1235c",
        "6115bd2c-957a-4543-abff-5fae52992ff2",
    ]:
        print(f"Filesystem ID {FILESYSTEM_RESOURCE_ID} not supported. Please adjust config_alcf.py.")
        sys.exit(1)

    # Filesystem/Compute compatibility
    if COMPUTE_RESOURCE_ID == "0325fc07-6fb7-4453-b772-3d5030b2df72":
        if FILESYSTEM_RESOURCE_ID == "6115bd2c-957a-4543-abff-5fae52992ff2":
            print(f"Home is not supported with Aurora yet. Please use Flare.")
            sys.exit(1)
        if FILESYSTEM_RESOURCE_ID == "1c3ad9d4-2e91-42bc-becb-72b1fde1235c":
            print(f"Eagle is not mounted on Aurora. Please use Flare.")
            sys.exit(1)
    if COMPUTE_RESOURCE_ID in ["55c1c993-1124-47f9-b823-514ba3849a9a", "8b9b42f7-572a-4909-8472-a0453436304c"]:
        if FILESYSTEM_RESOURCE_ID == "154bb3be-5d12-4a76-a16b-898b8e310a4b":
            print(f"Flare is not mounted on Polarix/Crux. Please use Eagle or Home.")
            sys.exit(1)

    # ALCF username and project
    if not alcf_username:
        print(f"'alcf_username' must be defined. Please adjust config_alcf.py.")
        sys.exit(1)
    if FILESYSTEM_RESOURCE_ID != "6115bd2c-957a-4543-abff-5fae52992ff2":
        if not alcf_project:
            print(f"'alcf_project' must be defined when targetting Eagle or Flare. Please adjust config_alcf.py.")
            sys.exit(1)


