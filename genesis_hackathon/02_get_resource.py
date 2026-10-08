"""
Script to query the status of a single resource given its ID.
TODO: Pass the resource ID when calling this script.
"""

import argparse
import json
import requests

from models import BaseConfig, Facilities
from utils import get_base_config


# Query the status of a specific resource
def get_resource(config: BaseConfig, resource_id):
    response = requests.get(f"{config.base_url}/status/resources/{resource_id}")
    return json.dumps(response.json(), indent=2)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "resource_id",
        help="Resource ID"
    )
    parser.add_argument(
        "--facility",
        required=True,
        choices=Facilities,
        help="Facility to query",
    )
    args = parser.parse_args()

    print(get_resource(get_base_config(args.facility), args.resource_id))
