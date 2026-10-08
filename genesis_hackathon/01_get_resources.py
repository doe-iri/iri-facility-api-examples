"""
Script to query the status of all resources.
Optional argument to extract a resource based on its name.
"""

import argparse
import json
import requests

from models import BaseConfig, Facilities
from utils import get_base_config


def get_resources(config: BaseConfig, resource_name: str = None):

    # Query the status of all resources
    response = requests.get(f"{config.base_url}/status/resources")
    resources = response.json()

    # Filter to extract a resource based on its name
    if resource_name is not None:
        resources = [r for r in resources if r["name"].lower() == resource_name.lower()]
        
    return json.dumps(resources, indent=2)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "resource_name",
        nargs="?",
        help="Resource name",
    )
    parser.add_argument(
        "--facility",
        required=True,
        choices=Facilities,
        help="Facility to query",
    )
    args = parser.parse_args()

    print(get_resources(get_base_config(args.facility), resource_name=args.resource_name))
