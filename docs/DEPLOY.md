# Deploying to Google Cloud Run

You run every command in this file yourself. Nothing here has been run for you,
and `gcloud` is not installed on the machine this was written on, so **the
commands are untested**. Before relying on a flag, check it with `--help`
(for example `gcloud run deploy --help`), because gcloud flags change.

What this sets up: the Docker image is built by Google Cloud Build, stored in
Artifact Registry, and run on Cloud Run. Cloud Run starts the container when a
request arrives and scales to zero when idle.

## 1. Decisions to make first

| Decision | Recommendation | Why |
|---|---|---|
| Public or private service | Private until you have read the licence notes in `docs/MODEL_CARD.md` | A public URL lets anyone call models derived from a non-commercial dataset, and they run on your bill. Private needs an identity token for every call, which the demo page cannot send. For a public demo, use `--allow-unauthenticated` below, and only after you decide. |
| Region | `australia-southeast1` (Sydney) | The nearer of the two Australian regions to Wellington. `australia-southeast2` (Melbourne) also works. There is no New Zealand Cloud Run region that I could confirm, so check the region list in the console. |
| Project | A new project used only for this | Deleting the project at the end removes everything and stops all charges. |

## 2. One-time setup

Install the CLI and sign in:

```bash
brew install --cask google-cloud-sdk
gcloud auth login
```

Set your values. Edit the first three lines.

```bash
export PROJECT_ID="chess-detection-demo-CHANGE-ME"   # must be globally unique
export BILLING_ACCOUNT="XXXXXX-XXXXXX-XXXXXX"        # from: gcloud billing accounts list
export REGION="australia-southeast1"
export REPO="chess"
export SERVICE="chess-api"
export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$SERVICE:v1"
```

Create the project, link billing, and turn on the services used:

```bash
gcloud projects create "$PROJECT_ID"
gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT"
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
    cloudbuild.googleapis.com billingbudgets.googleapis.com
gcloud artifacts repositories create "$REPO" --repository-format=docker \
    --location="$REGION" --description="chess detection images"
```

## 3. Set a budget alert before anything else

A budget sends emails as spend passes a threshold. **It does not stop spending.**
Create one now, in your billing account's currency:

```bash
gcloud billing budgets create --billing-account="$BILLING_ACCOUNT" \
    --display-name="chess-demo" --budget-amount=20 \
    --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 \
    --threshold-rule=percent=1.0
```

The same thing is in the console under Billing, then Budgets and alerts. The
real limits on cost are the instance cap and the teardown in section 7.

## 4. Build and push the image

Run this from the repo root, on a checkout that has `models/synthetic.pt` and
`models/finetuned.pt`:

```bash
gcloud builds submit --tag "$IMAGE" .
```

This uploads your source to Cloud Build, which builds on an amd64 machine and
pushes to Artifact Registry. That matters because an image built on an
Apple-silicon Mac is arm64 and will not run on Cloud Run. If you would rather
build locally, use `docker buildx build --platform linux/amd64 ...`, then push.

`.gcloudignore` controls what is uploaded. It must stay, because without it
gcloud uses `.gitignore`, which excludes `*.pt`, and the build would have no
weights.

### How the weights get into the image, and the trade-off

The two weight files are copied into the image at build time. They stay out of
git, and only your private Artifact Registry holds the image.

- **Good:** nothing to download at startup, no storage bucket to manage, and a
  given image always contains the same models.
- **Bad:** changing a model means rebuilding the image, and the image is about
  2 GB, which slows cold starts.
- **Do not:** make the Artifact Registry repository public or share the image.
  That would distribute the weights, which the licence notes advise against.

The alternatives are a Cloud Storage bucket the container reads at startup, or
Hugging Face or a GitHub Release. Either needs a licence decision first.

## 5. Deploy

```bash
gcloud run deploy "$SERVICE" \
    --image "$IMAGE" --region "$REGION" \
    --min-instances 0 --max-instances 2 \
    --memory 2Gi --cpu 1 --concurrency 2 --timeout 60 \
    --cpu-boost --port 8080 \
    --allow-unauthenticated
```

Delete the last line to keep the service private. What each setting is for:

| Setting | Value | Reason |
|---|---|---|
| `--min-instances 0` | scale to zero | No cost while idle. The price is a slow first request after a quiet period. |
| `--max-instances 2` | small cap | Limits how much a burst of traffic can cost. |
| `--memory 2Gi` | 2 GiB | Measured locally in Docker: about 321 MiB right after start and 808 MiB after six predictions on a 3024 pixel photo. 1 GiB would leave little headroom, 2 GiB leaves plenty. Cloud Run itself is untested. |
| `--cpu 1` | one vCPU | Inference runs on CPU. |
| `--concurrency 2` | two requests per instance | With one CPU, more parallel requests mostly queue. |
| `--timeout 60` | 60 s | Enough for a cold start plus one prediction. |
| `--cpu-boost` | extra CPU during startup | Helps the model load faster after a cold start. |

The command prints the service URL. Test it:

```bash
URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')
curl "$URL/health"
curl -X POST "$URL/predict?model=finetuned" -F "file=@board.jpg;type=image/jpeg"
```

### Cold starts

With zero minimum instances, the first request after idle time waits while the
container starts and loads both models. I have not measured how long that takes
on Cloud Run, so it is not stated here (TBD). The demo page asks `/health` as
soon as it loads, which starts the container before you upload. A minimum of one
instance removes the wait but costs money all the time.

## 6. Connect the demo page

The API only accepts browser requests from origins listed in `CORS_ORIGINS`.
Set it to the address where you host the page, with no trailing slash:

```bash
gcloud run services update "$SERVICE" --region "$REGION" \
    --update-env-vars CORS_ORIGINS="https://YOUR-DEMO-ORIGIN"
```

For more than one origin, gcloud needs a custom separator because the value
contains commas:

```bash
gcloud run services update "$SERVICE" --region "$REGION" \
    --update-env-vars "^@^CORS_ORIGINS=https://one.example,https://two.example"
```

Then open the demo page with the API address in the link, for example
`https://YOUR-DEMO-ORIGIN/?api=<service URL>`. Hosting the page (GitHub Pages is
one option) is not set up.

## 7. Tear everything down

The simplest and most complete way is to delete the project, which removes the
service, the images and the build files:

```bash
gcloud projects delete "$PROJECT_ID"
```

Google keeps a deleted project recoverable for a period before removing it for
good. If you want to keep the project, delete the parts instead:

```bash
gcloud run services delete "$SERVICE" --region "$REGION"
gcloud artifacts repositories delete "$REPO" --location="$REGION"
```

Cloud Build also leaves uploaded source in a storage bucket in the project;
delete it in the console under Cloud Storage. Afterwards, check that nothing is
left running:

```bash
gcloud run services list
gcloud artifacts repositories list
```

Also delete the budget (Billing, then Budgets and alerts) if you no longer
want its emails.

## Not verified

- No command in this file has been run.
- The amd64 image has not been built, so its size may differ from the 2.01 GB
  measured for the arm64 image.
- Cold-start time and per-request latency on Cloud Run have not been measured.
- The budget flags and the region list should be checked against current docs.
