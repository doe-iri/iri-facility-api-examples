# Genesis Mission Hackathon

The DOE IRI APIs allow you to programmatically execute jobs on HPC systems, trigger filesystem operations, and view your current allocations. This demo focuses on the execution of simple jobs to demonstrate the overall capabilities.

## 1. Environment

In your environment of choice, install the dependencies for the exercises:
```
pip install -r requirements.txt
```

Create a `.env` file to store your IRI tokens:
```bash
IRI_TOKEN_ALCF=""
IRI_TOKEN_NERSC=""
AMSC_TOKEN=""  # ESnet; or set AMSC_TOKEN_FILE (default /tmp/amsc-token.txt)
IRI_TOKEN_OLCF="" # OLCF S3M token
OLCF_S3M_PROJECT="" # OLCF only, set to project id (e.g., abc123) used to create S3M token
```

Make sure you edit the `config_<facility>.py` files to set your exercises.


## 2. Get Your IRI Token

### ALCF

Install the ALCF token manager and authenticate with Globus Auth:
```bash
pip install alcf-tokens
alcf-tokens login iri
```

Copy the URL to your browser, authenticate with your ALCF credentials, and copy-paste the authorization code back to your terminal.

Print your IRI API token:
```bash
alcf-tokens get-token iri
```

### NERSC

Install the NERSC token manager and authenticate with Globus Auth:
```bash
pip install nersc-tokens
nersc-tokens login iri
```

Open the URL, sign in with your NERSC account, approve access, and paste the authorization code.

Print your IRI API token:
```bash
nersc-tokens get-token iri
```

### OLCF

To access the IRI API via the OLCF Secure Scientific Service Mesh (S3M), go to [https://my.olcf.ornl.gov/](https://my.olcf.ornl.gov/), select ‘Moderate‘, and hit the ‘Log In‘ button.

- Enter your OLCF username
- Enter your Passcode (4 digit PIN + 6 digit RSA Token)
- Select your project from the ‘My Projects’ drop down top-left
- Select ‘S3M Access → New Token’ in left sidebar
- Choose ‘AmSC IRI API’ token permissions, give your token a name (e.g., ‘myuser-Oct15’), select ‘1 week’ for token lifetime, then hit ‘Submit’
- Copy the S3M token string using the provided button and save it for later use
(export IRI_TOKEN_OLCF=<S3M_TOKEN> or add IRI_TOKEN_OLCF to the `.env` file.)

### ESnet

Login to My AmSC website: [https://my.amsc.energy.gov/](https://my.amsc.energy.gov/) and select a project in the left sidebar. The PAT is bound to this project. Make sure "Genesis Mission October 2026 Hackathon" is selected.

- Click Personal Access Tokens and click New
- Enter token description and then hit Submit
- Copy the AmSC token using the provided button and save it for later use:
  - export AMSC_TOKEN=<YOUR AMSC TOKEN>
  - save into a file /tmp/amsc_token.txt
  - or add AMSC_TOKEN to the `.env` file.


### Documentation
- ALCF
  - v1: [https://docs.alcf.anl.gov/services/iri-api/#getting-your-api-token](https://docs.alcf.anl.gov/services/iri-api/#getting-your-api-token)
- NERSC
  - v1: [https://github.com/NERSC/iri-api-get-globus-token](https://github.com/NERSC/iri-api-get-globus-token)
  - v2: [https://docs.nersc.gov/services/sfapi/authentication/](https://docs.nersc.gov/services/sfapi/authentication/)
- ESnet (Use v2 for quick access):
  - v1: (SLOW) Globus token. [https://github.com/doe-iri/iri-facility-api-examples/blob/main/login-globus.ipynb]. Send an email to jbalcas@es.net with introspection
  - v2: (FAST) AmSC token (PAT) from MyAmSC for the `hackathon2610` project, see `../login-amsc.ipynb`. ESnet accepts AmSC tokens on `api/v2` only.
- OLCF
  - v2: [https://docs.olcf.ornl.gov/services_and_applications/s3m/overview.html#get-a-token](https://docs.olcf.ornl.gov/services_and_applications/s3m/overview.html#get-a-token)


## 3. Main Exercises

Once your `config_<facility>.py` files are set, you only need to execute the exercise files with the `--facility` flag to target a facility.

### 3.a. View Resources and their Status

Execute the following script to view the metadata of resources:
```bash
python 01_get_resources.py --facility <select-facility-here>
```

You can filter the list by adding the resource name as an argument.
```bash
python 01_get_resources.py <resource-name> --facility <select-facility-here>
```

For each resource, `current_status` reports whether the resource is *up* and ready to use, and `id` uniquely identifies the resource. To query a specific resource from its ID without going through a list, execute the following:

```bash
python 02_get_resource.py <select-resource-id> --facility <select-facility-here>
```

### 3.b. Submit Jobs

Submit a job with:
```bash
python 03_submit_job.py --facility <select-facility-here>
```

If successful, the above command should return the job ID (**keep this ID for the following steps**).
```json
{
  "id": "<your-job-id>",
  "status": {
    "state": "queued",
    "exit_code": 0
  }
}
```

The job will execute the content of the `COMMANDS` field defined in the `config_<facility>.py` files. If you kept the default content, the job will run on 1 node and write your username and the compute node hostname in the `STDOUT_PATH` file.

Execute the following to query the state of your job:
```bash
python 04_get_job_state.py <job-id> --facility <select-facility-here>
```

Once your job is `completed` or `failed`, continue to the next section.

### 3.c. View Job Results

You can view the result of your jobs with Filesystem operations. If your PBS job completed, execute the following:
```bash
python 05_view_file.py <absolute-path-to-stdout-file> --facility <select-facility-here>
```

If your PBS job failed, execute the following:
```bash
python 05_view_file.py <absolute-path-to-stderr-file> --facility <select-facility-here>
```

All filesystem operations are asynchronous, meaning you will always get back a `task_id` when using the Filesystem component. The `05_view_file.py` script automatically checks the status of your task in a loop until it is completed. 

### 3.d. Query Lists of Jobs

Execute the following to query a list of jobs:
```bash
python 06_list_jobs.py --facility <select-facility-here>
```
You can filter the list by editing the `FILTERS` field in `config_<facility>.py`.

### 3.e. Cancel Job

The IRI API allows you to cancel jobs that are already submitted to the scheduler. First, incorporate a longer sleep (`sleep 30`) in your `COMMANDS` field to give yourself some time to cancel the job. Then, submit the job with
```bash
python 03_submit_job.py --facility <select-facility-here>
```

Cancel your job with your job ID by executing:
```bash
python 07_cancel_job.py <job-id> --facility <select-facility-here>
```

Follow the state of your job until it is labeled as `canceled`.
```bash
python 04_get_job_state.py <job-id> --facility <select-facility-here>
```

### 3.f. Query Allocations

Execute the following script to view your active projects:
```bash
python 08_get_projects.py --facility <select-facility-here>
```

You can filter the list by adding the project name as an argument:
```bash
python 08_get_projects.py <project-name> --facility <select-facility-here>
```

To query a specific project from its ID without going through a list, execute the following:

```bash
python 09_get_project.py <project-id> --facility <select-facility-here>
```

To view the allocations tied to a project ID, execute the following:
```bash
python 10_get_allocations.py <project-id> --facility <select-facility-here>
```
