
# IRI Facility API Examples (Notebooks)

This repo contains **Jupyter notebooks** that demonstrate an end-to-end workflow against the IRI API:

1. Authenticate (Globus or Facility specific authentication)
2. Use the Filesystem API to list/download/upload files
3. Submit Compute jobs
4. Collect logs (stdout/stderr) and list generated artifacts (e.g., MiniGPT outputs)

---

## Contents

- `start-notebook.sh` — creates a local `.venv`, installs Jupyter + ipykernel, registers a kernel, then starts Jupyter Notebook
- `login-globus.ipynb` — get an IRI API token via Globus (for endpoints that support Globus auth. THIS IS TEMPORARY AND WILL NOT BE SUPPORTED IN THE FUTURE. Currently supported by NERSC and ESnet IRI Endpoints
- `login-esnet.ipynb` — get an IRI API token for ESnet IRI Endpoints (Facility Specific)
- `login-alcf.ipynb` - get an IRI API token for ALCF IRI Endpoints (Facility Specific)
- `filesystem.ipynb` — list/download/upload/check paths via the IRI Filesystem API
- `compute-jobs.ipynb` — compute job examples (new compute payload format)
- `compute-job-mini-gpt.ipynb` — MiniGPT training job example using container image + shared storage

> The compute + filesystem notebooks assume your shared storage is available at:
> `/data/home/<username>` (example: `/data/home/jbalcas`). Modify as needed.

---

## Prerequisites

- Python 3.10+ recommended
- Credentials for **one** authentication method that is supported by the Facility:
  - Globus OAuth client credentials
  - ESnet/SENSE credentials
  - A pre-minted `IRI_API_TOKEN` in the environment

---

## Quickstart

### 1) Create `.env`

Create a file named `.env` in the repo root directory:

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
- Launch Jupyter Notebook

---

## Notebook Workflow

### Step 1 — Authenticate

Choose one:

#### Globus (For NERSC and ESnet Endpoints)

Run:

```
login-globus.ipynb
```

#### ESnet / SENSE (Facility Specific)

```
login-esnet.ipynb
```

#### ALCF (Facility Specific)

```
login-alcf.ipynb
```


#### Manual token

```
export IRI_API_TOKEN="REPLACEME"
```

---

### Step 2 — Filesystem API

Open:

```
filesystem.ipynb
```

Use it to:

- list files
- download files
- upload test files
- verify the shared path (`DEFAULT_JOB_DIR`)

---

### Step 3 — Compute Jobs

Open:

```
compute-jobs.ipynb
```

This notebook demonstrates compute job submission (without containers) and allow to specify:

- executable
- arguments
- resources
- queue + account
- stdout / stderr capture

---

### Step 4 — MiniGPT Training Demo

Open:

```
compute-job-mini-gpt.ipynb
```

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
