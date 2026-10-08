import sys

from models import Config


def get_config(facility: str) -> Config:
    """Return the config for the target facility."""

    facility = facility.lower()

    # Import only the selected config: each config exits when its own token is missing.
    if facility.lower() == "alcf":
        import config_alcf
        return config_alcf.config
    elif facility.lower() == "nersc":
        import config_nersc
        return config_nersc.config
    elif facility.lower() == "esnet":
        import config_esnet
        return config_esnet.config
    elif facility.lower() == "olcf-open":
        import config_olcf_open
        return config_olcf_open.config
    elif facility.lower() == "olcf-moderate":
            import config_olcf_moderate
            return config_olcf_moderate.config
    else:
        print(f"Facility {facility} not supported yet.")
        sys.exit(1)


def get_headers(token: str) -> dict[str, str]:
    """Return authenticated request headers."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
