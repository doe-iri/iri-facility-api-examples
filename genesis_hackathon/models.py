from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class Facilities(str, Enum):
    alcf = "alcf"
    nersc = "nersc"
    esnet = "esnet"
    olcf_open = "olcf-open"
    olcf_moderate = "olcf-moderate"
    

class Config(BaseModel):

    # API URL
    base_url: str = Field(min_length=1)
    token: str = Field(min_length=1)

    # Job submission resources
    compute_resource_id: str = Field(min_length=1)
    filesystem_resource_id: str = Field(min_length=1)

    # Job submission parameters
    nodes: int = Field(ge=1)
    walltime_sec: int = Field(ge=300)
    queue: str = Field(min_length=1)
    compute_allocation: str = Field(min_length=1)
    working_directory: Optional[str] = Field(default=None)
    stdout_path: str = Field(min_length=1)
    stderr_path: str = Field(min_length=1)
    custom_attributes: Optional[dict] = Field(default={}) 
    commands: str = Field(min_length=1)
    
    # Job list
    filters: Optional[dict] = None