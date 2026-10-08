"""
Script to view the content of a file on a filesystem.
TODO: Pass the absolute path of the file when calling this script.
"""

import argparse
import json
import requests
from time import sleep

from models import Config, Facilities
from utils import get_config, get_headers


# Submit filesystem operation and get back a task ID
def submit_view_file(config: Config, file_path: str) -> str:

    # Submit view command
    print("\n=========================")
    print("SUBMIT FILESYSTEM COMMAND")
    print("=========================\n")
    print(f"Submitting filesystem view command to {config.filesystem_resource_id} ...")

    if "api/v1" in config.base_url:
        response = requests.get(
            f"{config.base_url}/filesystem/view/{config.filesystem_resource_id}",
            params={
                "path": file_path,
                "size": 1000,
                "offset": 0
            },
            headers=get_headers(config.token)
        )
    else:
        response = requests.post(
            f"{config.base_url}/filesystem/view/{config.filesystem_resource_id}",
            json={
                "path": file_path,
                "size": 1000,
                "offset": 0
            },
            headers=get_headers(config.token)
        )

    # Print task status
    response = response.json()
    print(json.dumps(response, indent=2))

    # Return task ID
    return response.get("task_id")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path",
        help="Absolute path of the file"
    )
    parser.add_argument(
        "--facility",
        required=True,
        choices=Facilities,
        help="Facility to query",
    )
    args = parser.parse_args()

    # Load config
    config = get_config(args.facility)

    # Submit filesytem operation and get back a task ID
    task_id = submit_view_file(config, args.file_path)

    print("\n==============")
    print("EXTRACT RESULT")
    print("==============\n")
    print(f"Waiting for filesystem task {task_id} to complete ...")

    while True:

        # Query task status every 2 seconds
        sleep(2)
        response = requests.get(
            f"{config.base_url}/task/{task_id}",
            headers=get_headers(config.token)
        )
        response = response.json()

        # Report task status
        task_status = response.get("status")
        print(f"Current status: {task_status}")

        # Exit loop if needed
        if task_status not in ["pending", "active"]:
            print()
            break

    # Print error or file content
    print(json.dumps(response.get("result"), indent=2))
    