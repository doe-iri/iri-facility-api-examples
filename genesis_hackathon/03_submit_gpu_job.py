"""
Submit a GPU job to a compute resource and get the job ID back.
"""

import argparse
import json
import requests

from models import Config, Facilities
from utils import get_config, get_headers


# Commands executed in the GPU job: identify the GPUs visible to the job
GPU_COMMANDS = """
echo Start
hostname
echo "=== GPU visibility ==="
env | grep -E '_VISIBLE_DEVICES=' || echo "No *_VISIBLE_DEVICES variable set"
# OLCF puts rocm-smi on PATH through the rocm module
command -v module >/dev/null 2>&1 && module load rocm >/dev/null 2>&1
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "=== nvidia-smi ==="
    nvidia-smi
    echo "=== GPU summary ==="
    nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version --format=csv
elif command -v rocm-smi >/dev/null 2>&1; then
    echo "=== rocm-smi ==="
    rocm-smi
    echo "=== GPU summary ==="
    rocm-smi --showproductname
else
    # No vendor tool: fall back to what the driver and devices expose
    echo "=== Basic GPU info (nvidia-smi and rocm-smi not found) ==="
    ls -l /dev/nvidia* /dev/kfd /dev/dri 2>/dev/null || echo "No GPU device files found"
    cat /proc/driver/nvidia/gpus/*/information 2>/dev/null
    command -v lspci >/dev/null 2>&1 && lspci | grep -i -E 'vga|3d|display'
fi
echo End
"""

# ESnet runs every job in a container: use an image that ships the CUDA tools
ESNET_GPU_IMAGE = "nvidia/cuda:12.4.1-base-ubuntu22.04"


# Submit GPU job to compute resource
def submit_gpu_job(config: Config, facility: Facilities, gpus: int):
    custom_attributes = dict(config.custom_attributes or {})
    if facility == Facilities.nersc:
        # Perlmutter places a job on GPU nodes only with the "gpu" constraint
        custom_attributes["constraint"] = "gpu"

    payload = {
        "executable": "/bin/bash",
        "arguments": ["-lc", GPU_COMMANDS],
        "name": "my-gpu-job",
        "stdout_path": config.stdout_path,
        "stderr_path": config.stderr_path,
        "resources": {
            "node_count": config.nodes,
            "gpu_cores_per_process": gpus,
        },
        "attributes": {
            "duration": config.walltime_sec,
            "queue_name": config.queue,
            "account": config.compute_allocation,
            "custom_attributes": custom_attributes,
        }
    }
    if facility == Facilities.esnet:
        payload["container"] = {"image": ESNET_GPU_IMAGE}
    if config.working_directory:
        payload["directory"] = config.working_directory
    if config.reservation:
        payload["attributes"]["reservation_id"] = config.reservation

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
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to request for the job process (default: 1)",
    )
    args = parser.parse_args()
    # The IRI API rejects 0: every resource count must be at least 1
    if args.gpus < 1:
        parser.error("--gpus must be at least 1")

    config = get_config(args.facility)
    print(submit_gpu_job(config, Facilities(args.facility), args.gpus))

    print()
    print("If job submission is successful, your job logs are set to:")
    print(config.stdout_path if config.stdout_path else "'stdout_path' not specified.")
    print(config.stderr_path if config.stderr_path else "'stderr_path' not specified.")
    print()
