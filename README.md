# IRI Facility API Examples (Notebooks)

This repo contains **Jupyter notebooks** that demonstrate an end-to-end workflow against the IRI API:

1. Authenticate (AmSC PAT via RIG, Globus, or Facility specific authentication)
2. Use the Filesystem API to list/download/upload files
3. Submit Compute jobs
4. Collect logs (stdout/stderr) and list generated artifacts (e.g., MiniGPT outputs)

There are two ways to reach a facility:

- **Directly with a facility token** (`iri/`). You call each facility's IRI API with a token issued for that facility.
- **Through AmSC with an AmSC PAT** (`amsc/`). One AmSC Personal Access Token for all facilities: calls go through RIG
  (Resource Integration Gateway), which attaches the facility credential for your project.

---

## Contents

```
.
├── start-notebook.sh              # create .venv, install Jupyter, start it
├── mini-gpt/                      # MiniGPT container image sources
├── iri/                           # direct facility access with a facility token -> ~/.iri_token.json
│   ├── login-globus.ipynb
│   ├── login-esnet.ipynb
│   ├── login-alcf.ipynb
│   ├── filesystem.ipynb           # IRI Filesystem API (v1)
│   ├── filesystem-v2.ipynb        # IRI Filesystem API (v2)
│   ├── compute-jobs.ipynb         # IRI compute + filesystem smoke test (v1)
│   ├── compute-jobs-v2.ipynb      # IRI compute + filesystem smoke test (v2)
│   ├── compute-jobs-alcf.ipynb    # same smoke test, ALCF payload
│   ├── compute-job-gpu-test.ipynb # GPU job + CUDA test
│   └── compute-job-mini-gpt.ipynb # MiniGPT training job (container)
└── amsc/                          # AmSC PAT + RIG notebooks -> ~/.amsc_token.json
    ├── login-amsc.ipynb
    ├── compute-jobs-amsc.ipynb
    ├── amsc-facility-access.ipynb
    ├── images/rig-ui/             # RIG-UI screenshots used by the AmSC notebooks
    └── globus/
        ├── rig_globus_tutorial.ipynb
        └── README.md
```

### Direct facility access

- [`start-notebook.sh`](start-notebook.sh) — creates a local `.venv`, installs Jupyter + ipykernel, registers a kernel, then starts Jupyter Notebook
- [`iri/login-globus.ipynb`](iri/login-globus.ipynb) — get an IRI API token via Globus (for endpoints that support Globus auth). THIS IS TEMPORARY AND WILL NOT BE SUPPORTED IN THE FUTURE. Currently supported by NERSC and ESnet IRI Endpoints
- [`iri/login-esnet.ipynb`](iri/login-esnet.ipynb) — get an IRI API token for ESnet IRI Endpoints (Facility Specific)
- [`iri/login-alcf.ipynb`](iri/login-alcf.ipynb) — get an IRI API token for ALCF IRI Endpoints (Facility Specific)
- [`iri/filesystem.ipynb`](iri/filesystem.ipynb) — list/download/upload/check paths via the IRI Filesystem API (v1)
- [`iri/filesystem-v2.ipynb`](iri/filesystem-v2.ipynb) — the same filesystem test for the IRI **v2** API
- [`iri/compute-jobs.ipynb`](iri/compute-jobs.ipynb) — compute job examples (new compute payload format, v1)
- [`iri/compute-jobs-v2.ipynb`](iri/compute-jobs-v2.ipynb) — the same compute + filesystem smoke test for the IRI **v2** API
- [`iri/compute-jobs-alcf.ipynb`](iri/compute-jobs-alcf.ipynb) — the same compute + filesystem smoke test with the ALCF payload adjustments
- [`iri/compute-job-gpu-test.ipynb`](iri/compute-job-gpu-test.ipynb) — requests a GPU for a job and tests CUDA
- [`iri/compute-job-mini-gpt.ipynb`](iri/compute-job-mini-gpt.ipynb) — MiniGPT training job example using container image + shared storage ([`mini-gpt/`](mini-gpt/) builds the image)

The `iri/` login notebooks save the token to `~/.iri_token.json` as `IRI_API_TOKEN`; the other `iri/` notebooks read it
from there. The same token works for v1 and v2.

#### IRI API v1 and v2

Facilities serve the IRI Facility API under `/api/v1` and, increasingly, `/api/v2`:

| Facility | v1 | v2 |
|---|---|---|
| ESnet East / West | yes | yes |
| NERSC | yes | yes |
| ALCF | yes | no |
| OLCF (S3M, via AmSC) | | status + compute only |

The `-v2` notebooks are written for v2. What changes compared with v1:

- Filesystem reads (`ls`, `download`, `stat`, `head`, `tail`, `view`, `checksum`, `file`), `chmod`/`chown` and `rm` are
  `POST` with a JSON body instead of `GET`/`PUT`/`DELETE` with query parameters (same field names).
- `cp`/`mv`/`compress`/`extract` take the source as `path`; compression is a URN (`urn:doe-iri:compression:gzip`).
- `resource_type` and other enumerations are URNs (e.g. `urn:doe-iri:resource:compute:system`); resources advertise
  their operations as HAL `_links` (e.g. `iri:submit-job`, `iri:list-directory`).
- New: `GET /account/whoami` (your facility username), `GET /storage/locations/{resource_id}` (your home/project/scratch
  paths), and an optional `Idempotency-Key` header for job submission.

> The compute + filesystem notebooks assume your shared storage is available at:
> `/data/home/<username>` (example: `/data/home/jbalcas`). Modify as needed.

### AmSC access through RIG

- [`amsc/login-amsc.ipynb`](amsc/login-amsc.ipynb) — log in to MyAmSC, select a project, generate an AmSC Personal Access Token (PAT), and save it to `~/.amsc_token.json`
- [`amsc/compute-jobs-amsc.ipynb`](amsc/compute-jobs-amsc.ipynb) — the same compute + filesystem smoke test with an AmSC PAT, through RIG (discovers facilities via RIG; handles IRI v1 and v2)
- [`amsc/amsc-facility-access.ipynb`](amsc/amsc-facility-access.ipynb) — what each facility needs before the PAT works through RIG (allocations, RIG-UI Credential Vault setup, with screenshots), plus an access check
- [`amsc/globus/rig_globus_tutorial.ipynb`](amsc/globus/rig_globus_tutorial.ipynb) — Globus transfers through RIG with the AmSC PAT (your project's Globus Service Account and registered collections); `curl` version in [`amsc/globus/README.md`](amsc/globus/README.md)

---

## Prerequisites

- Python 3.10+ recommended
- Credentials for **one** authentication method that is supported by the Facility:
  - An AmSC account with a project (for the AmSC PAT + RIG notebooks in `amsc/`)
  - Globus OAuth client credentials
  - ESnet/SENSE credentials
  - A pre-minted `IRI_API_TOKEN` in the environment

---

## Quickstart

### 1) Create `.env`

Create a file named `.env` in the repo root directory. Notebooks in `iri/` and `amsc/` find it too (it is looked up from
the notebook's directory upwards).

```dotenv
# Globus settings (if use globus auth)
GLOBUS_ID="REPLACEME"
GLOBUS_SECRET="REPLACEME"

# ESnet Auth settings (if use ESnet auth)
SENSE_AUTH_ENDPOINT="REPLACEME"
SENSE_CLIENT_ID="REPLACEME"
SENSE_SECRET="REPLACEME"
SENSE_USERNAME="REPLACEME"
SENSE_PASSWORD="REPLACEME"
SENSE_VERIFY_TLS="true"
SENSE_TIMEOUT=30

# Optional defaults for compute job
DEFAULT_JOB_DIR=/data/home/jbalcas
DEFAULT_QUEUE=debug
DEFAULT_ACCOUNT=interactive

# IRI API endpoint
IRI_BASE_URL=https://iri-dev.ppg.es.net/api/v1
# IRI_API_TOKEN=12345 Manual override

# AmSC PAT + RIG (optional; amsc/login-amsc defaults to the staging deployment)
# AMSC_ENV=staging
# AMSC_MYAMSC_URL=https://my.amsc.energy.gov/
# AMSC_RIG_URL=https://rig.staging.american-science-cloud.org
# AMSC_PAT=REPLACEME Manual override
# AMSC_FACILITY=esnet-east   # preselect a facility in amsc/compute-jobs-amsc
# IRI_USERNAME=jbalcas       # facility username, used for the default job directory

# IRI v2 notebooks (iri/*-v2.ipynb; all optional)
# IRI_USERNAME=jbalcas        # default: GET /account/whoami
# IRI_JOB_DIR=/data/home/jbalcas   # compute-jobs-v2 job directory; default: GET /storage/locations
# IRI_FS_TEST_DIR=/data/home/jbalcas   # filesystem-v2 test directory parent; default: GET /storage/locations
# IRI_QUEUE=debug
# IRI_ACCOUNT=interactive
# IRI_IDEMPOTENCY=true        # send an Idempotency-Key on job submission (v2)
```

---

### 2) Start Jupyter

Run:

```bash
bash start-notebook.sh
```

This will:

- Create `.venv` if missing
- Activate the environment
- Install `jupyter` and `ipykernel`
- Register kernel `iri-examples`
- Launch Jupyter Notebook in the repo root (open notebooks in `iri/` and `amsc/` from there)

---

## Notebook Workflow

### Step 1 — Authenticate

Choose one:

#### Globus (For NERSC and ESnet Endpoints)

Run [`iri/login-globus.ipynb`](iri/login-globus.ipynb).

#### ESnet / SENSE (Facility Specific)

Run [`iri/login-esnet.ipynb`](iri/login-esnet.ipynb).

#### ALCF (Facility Specific)

Run [`iri/login-alcf.ipynb`](iri/login-alcf.ipynb).

#### AmSC PAT (all facilities through RIG)

Run [`amsc/login-amsc.ipynb`](amsc/login-amsc.ipynb).

Opens MyAmSC: log in, select a project in the left sidebar, and under **Personal Access Tokens** click **New Token**. Paste
the PAT into the notebook; it is checked against RIG and saved to `~/.amsc_token.json`. The PAT is bound to the selected
project. Use it with the notebooks in [`amsc/`](amsc/).

Not every facility accepts the PAT on its own: ESnet and PNNL need resources for your project at the site; NERSC, ALCF and
ORNL/OLCF also need a facility credential in the RIG-UI Credential Vault. See
[`amsc/amsc-facility-access.ipynb`](amsc/amsc-facility-access.ipynb).

#### Manual token

```
export IRI_API_TOKEN="REPLACEME"
```

---

### Step 2 — Filesystem API

Open [`iri/filesystem.ipynb`](iri/filesystem.ipynb) (IRI v1) or [`iri/filesystem-v2.ipynb`](iri/filesystem-v2.ipynb) (IRI v2).

Use it to:

- list files
- download files
- upload test files
- verify the shared path (`DEFAULT_JOB_DIR`)

---

### Step 3 — Compute Jobs

Open [`iri/compute-jobs.ipynb`](iri/compute-jobs.ipynb) (IRI v1; [`iri/compute-jobs-alcf.ipynb`](iri/compute-jobs-alcf.ipynb) for ALCF)
or [`iri/compute-jobs-v2.ipynb`](iri/compute-jobs-v2.ipynb) (IRI v2: ESnet, NERSC).

For the AmSC PAT, open [`amsc/compute-jobs-amsc.ipynb`](amsc/compute-jobs-amsc.ipynb) instead. It asks RIG which
facilities you can reach, lets you pick one, and then runs the same steps through `https://<RIG>/rig/external/<facility>/...`.

This notebook demonstrates compute job submission (without containers) and allow to specify:

- executable
- arguments
- resources
- queue + account
- stdout / stderr capture

---

### Step 4 — MiniGPT Training Demo

Open [`iri/compute-job-mini-gpt.ipynb`](iri/compute-job-mini-gpt.ipynb).

This notebook:

1. Submits a container based job
2. Runs MiniGPT training
3. Writes logs to the job directory
4. Generates model artifacts

Example output:

```
/data/home/jbalcas/
 ├── minigpt_stdout_<timestamp>.log
 ├── minigpt_stderr_<timestamp>.log
 └── amsc-iri-demo-results-<timestamp>/
        tiny_gpt2_artifacts/
            tiny_gpt2_model/
                model.safetensors
                config.json
                tokenizer.json
```

---

### Step 5 — Globus transfers through RIG (AmSC PAT)

Open [`amsc/globus/rig_globus_tutorial.ipynb`](amsc/globus/rig_globus_tutorial.ipynb) (or follow the `curl` version in
[`amsc/globus/README.md`](amsc/globus/README.md)). It lists your project's registered Globus collections, then lists,
copies and deletes a file through RIG using your project's Globus Service Account.
