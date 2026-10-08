"""
Script to list PBS jobs on a compute resource.
"""

import argparse
import json
import requests

from models import Config, Facilities
from utils import get_config, get_headers

# Define query parameters
params = {
    "historical": "true", # "true" will include completed jobs
    "limit": 10, # maximum number of jobs returned
    "offset": 0,
}


# Submit request
def list_jobs(config: Config):
    response = requests.post(
        f"{config.base_url}/compute/status/{config.compute_resource_id}",
        params=params,
        json=config.filters,
        headers=get_headers(config.token),
    )
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

    print(list_jobs(get_config(args.facility)))