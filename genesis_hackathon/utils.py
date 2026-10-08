import sys

from models import BaseConfig, Config


def get_base_config(facility: str) -> BaseConfig:
    """Return a BaseConfig with only base_url for unauthenticated requests."""
    facility = facility.lower()

    if facility == "alcf":
        import config_alcf
        return BaseConfig(base_url=config_alcf.BASE_URL)
    elif facility == "nersc":
        import config_nersc
        return BaseConfig(base_url=config_nersc.BASE_URL)
    elif facility == "esnet":
        import config_esnet
        return BaseConfig(base_url=config_esnet.BASE_URL)
    elif facility == "olcf-open":
        import config_olcf_open
        return BaseConfig(base_url=config_olcf_open.BASE_URL)
    elif facility == "olcf-moderate":
        import config_olcf_moderate
        return BaseConfig(base_url=config_olcf_moderate.BASE_URL)
    else:
        print(f"Facility {facility} not supported yet.")
        sys.exit(1)


def get_config(facility: str) -> Config:
    """Return the full authenticated config for the target facility."""
    facility = facility.lower()

    if facility == "alcf":
        import config_alcf as c
        _require(c.TOKEN, "IRI_TOKEN_ALCF")
        c.validate_input()
    elif facility == "nersc":
        import config_nersc as c
        _require(c.TOKEN, "IRI_TOKEN_NERSC")
    elif facility == "esnet":
        import config_esnet as c
        _require(c.TOKEN, "AMSC_TOKEN (or AMSC_TOKEN_FILE)")
    elif facility == "olcf-open":
        import config_olcf_open as c
        _require(c.TOKEN, "IRI_TOKEN_OLCF")
        _require(c.OLCF_PROJECT, "OLCF_S3M_PROJECT")
    elif facility == "olcf-moderate":
        import config_olcf_moderate as c
        _require(c.TOKEN, "IRI_TOKEN_OLCF")
        _require(c.OLCF_PROJECT, "OLCF_S3M_PROJECT")
    else:
        print(f"Facility {facility} not supported yet.")
        sys.exit(1)

    return Config(
        base_url=c.BASE_URL,
        token=c.TOKEN,
        compute_resource_id=c.COMPUTE_RESOURCE_ID,
        filesystem_resource_id=c.FILESYSTEM_RESOURCE_ID,
        nodes=c.NODES,
        walltime_sec=c.WALLTIME_SEC,
        queue=c.QUEUE,
        compute_allocation=c.COMPUTE_ALLOCATION,
        working_directory=getattr(c, "WORKING_DIR", None),
        stdout_path=c.STDOUT_PATH,
        stderr_path=c.STDERR_PATH,
        commands=c.COMMANDS,
        filters=getattr(c, "FILTERS", {}),
        custom_attributes=getattr(c, "custom_attributes", {}),
    )


def _require(value, env_var: str):
    if not value:
        print(f"{env_var} missing in .env file.")
        sys.exit(1)


def get_headers(token: str) -> dict[str, str]:
    """Return authenticated request headers."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
