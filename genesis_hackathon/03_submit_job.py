"""
Submit a job to a compute resource and get the job ID back.
"""

import argparse
import json
import requests

from models import Config, Facilities
from utils import get_config, get_headers


# Submit job to compute resource
def submit_job(config: Config):
    payload = {
        "executable": "/bin/bash",
        "arguments": ["-lc", config.commands],
        "name": "my-job",
        "stdout_path": config.stdout_path,
        "stderr_path": config.stderr_path,
        "resources": {
            "node_count": config.nodes
        },
        "attributes": {
            "duration": config.walltime_sec,
            "queue_name": config.queue,
            "account": config.compute_allocation,
            "custom_attributes": config.custom_attributes,
        }
    }
    if config.working_directory:
        payload["directory"] = config.working_directory
    
    response = requests.post(
        f"{config.base_url}/compute/job/{config.compute_resource_id}",
        json=payload,
        headers=get_headers(config.token)
    )

    # Return job submission details with the job ID (or error if any) 
    return json.dumps(response.json(), indent=2)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--facility",
        required=True,
        choices=Facilities,
        help="Facility to query",
    )
    args = parser.parse_args()

    print(get_config(args.facility))
