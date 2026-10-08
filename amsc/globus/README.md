# RIG + Globus Transfer: end-to-end tutorial

Move data with Globus **through RIG** without your code ever handling a Globus token. You send your **AmSC PAT**; RIG
takes the project from the PAT (`amsc_project_context`), swaps the PAT for your **project's Globus Service Account**
token, and forwards the request unchanged to the Globus Transfer API.

```
You --(AmSC PAT)--> Kong --> RIG --(project Globus Service Account token)--> Globus Transfer API
```

Data moves between Globus collections (DTN to DTN), never through RIG. The Service Account can only reach the
collections registered for the project, under each registered path, with the registered permissions (`r` / `rw`).

Two ways to follow along:

- **Notebook**: [`rig_globus_tutorial.ipynb`](rig_globus_tutorial.ipynb), run top to bottom.
- **This README**: the same steps as `curl` commands.

Steps:

1. Load your AmSC PAT (from [`../login-amsc.ipynb`](../login-amsc.ipynb))
2. Discover your project's Globus collections
3. List files on a collection
4. Copy a file into a new directory on a collection
5. Delete the copy

> These are **real** Globus operations on your project's collections.

---

## Prerequisites

- An AmSC PAT from [`../login-amsc.ipynb`](../login-amsc.ipynb), saved to `~/.amsc_token.json`. The PAT's project is
  the project whose Service Account and collections are used.
- A **Globus Service Account provisioned for that project** by the VO / AmSC administrators.
- At least one **Globus collection registered for the project** in RIG-UI → **Globus Collections**
  (`https://rig-ui.staging.american-science-cloud.org/globus-collections`), with a small non-empty file under its path.
  To add your own guest collection: create it in the Globus web app, give the project's Service Account identity (shown on
  the Globus Collections page) read or read/write access, then register its UUID and path under
  **Add my own Globus endpoint**.

Notebook settings (environment or `.env`): `AMSC_PAT`, `AMSC_RIG_URL`, `AMSC_CREDENTIAL_VAULT_URL` (default to the values in
`~/.amsc_token.json`), plus `GLOBUS_SOURCE`, `GLOBUS_DEST` (`#`, collection UUID, or label) and `GLOBUS_FILENAME`.

---

## `curl` walkthrough

```bash
TOKEN=$(jq -r .AMSC_PAT ~/.amsc_token.json)
RIG=$(jq -r .AMSC_RIG_URL ~/.amsc_token.json)
TRANSFER="$RIG/globus/external/transfer/v0.10"
AUTH="Authorization: Bearer $TOKEN"
```

### 2) Discover your project's Globus collections

```bash
curl -s -H "$AUTH" "$RIG/globus/internal/collections" | jq
# -> {"project": ..., "service_account_identity": "<client-id>@clients.auth.globus.org",
#     "collections": [{"collection_id": ..., "normalized_path": ..., "permissions": "rw", ...}]}
SRC=<source collection_id>;       SRC_PATH=<its normalized_path>
DST=<destination collection_id>;  DST_PATH=<its normalized_path>   # needs "rw"
```

### 3) List files

```bash
curl -s -H "$AUTH" "$TRANSFER/operation/endpoint/$SRC/ls?path=$SRC_PATH" | jq '.DATA[] | {type, name, size}'
```

### 4) Copy a file

```bash
FILE=<file under SRC_PATH>
DIR="${DST_PATH%/}/amsc_rig_tutorial_$(date -u +%Y%m%d%H%M%S)"

curl -s -H "$AUTH" -H "Content-Type: application/json" \
  -d "{\"DATA_TYPE\": \"mkdir\", \"path\": \"$DIR\"}" "$TRANSFER/operation/endpoint/$DST/mkdir" | jq

SID=$(curl -s -H "$AUTH" "$TRANSFER/submission_id" | jq -r .value)
TASK=$(curl -s -H "$AUTH" -H "Content-Type: application/json" -d @- "$TRANSFER/transfer" <<EOF | jq -r .task_id
{"DATA_TYPE": "transfer", "submission_id": "$SID",
 "source_endpoint": "$SRC", "destination_endpoint": "$DST",
 "DATA": [{"DATA_TYPE": "transfer_item",
           "source_path": "${SRC_PATH%/}/$FILE", "destination_path": "$DIR/$FILE"}],
 "label": "AmSC RIG tutorial copy"}
EOF
)

# Poll until "SUCCEEDED":
curl -s -H "$AUTH" "$TRANSFER/task/$TASK" | jq '{status, bytes_transferred, nice_status_details}'
```

### 5) Delete the copy

```bash
SID=$(curl -s -H "$AUTH" "$TRANSFER/submission_id" | jq -r .value)
curl -s -H "$AUTH" -H "Content-Type: application/json" -d @- "$TRANSFER/delete" <<EOF | jq
{"DATA_TYPE": "delete", "submission_id": "$SID", "endpoint": "$DST", "recursive": true,
 "DATA": [{"DATA_TYPE": "delete_item", "path": "$DIR"}], "label": "AmSC RIG tutorial cleanup"}
EOF
```

---

## Troubleshooting

| Symptom | Meaning | Fix |
|---|---|---|
| 401 from RIG | PAT expired, revoked, or from another deployment | Run [`../login-amsc.ipynb`](../login-amsc.ipynb) again |
| 400 `Cannot determine caller's active project` | PAT has no project | Select a project in MyAmSC and create a new PAT |
| 409 `No Globus Service Account provisioned for project ...` | Project has no Globus Service Account | Ask the VO / AmSC administrators to provision one |
| 500 `Service Account credential problem` | The project's Service Account secret is invalid or missing | Ask the VO / AmSC administrators |
| Globus `PermissionDenied` / `ClientError.Forbidden` | Path outside the registered collection path, or the Service Account lost access | Stay under the collection's `normalized_path`; check the grant in Globus and in RIG-UI → Globus Collections |
| Transfer `FAILED` | Globus task-level error (e.g. source file missing) | Check `nice_status_details` |
