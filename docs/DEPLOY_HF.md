# Deploying to a private Hugging Face Space

You run every step below yourself. Nothing here has been run against Hugging
Face. What was checked locally: `scripts/prepare_space.sh` builds the upload
folder, the Docker image builds from that folder, and the container answers
`/health` as UID 1000. The Hugging Face side (creating the Space, uploading,
building, the token calls) is untested, so check anything odd against the
current Hugging Face docs.

## Why this option

A Docker Space runs the same Dockerfile as the rest of the project. The free
CPU tier is listed as 2 vCPU, 16 GB RAM and 50 GB of non-persistent disk, and
free Spaces sleep after 48 hours without visits and wake on the next one. The
service used about 808 MiB in local tests. These figures come from the Hugging
Face docs at the time of writing and can change.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Visibility | **Private** | The fine-tuned weights come from ChessReD (CC BY-NC-SA 4.0) and the synthetic weights come from a dataset whose licence is still TBD. A public Space would distribute both. |
| Cost | Free CPU Basic | No billing account needed. |

A private Space means only you can open it. Calls to the API need a Hugging
Face access token with read permission, sent as `Authorization: Bearer hf_...`.
The static demo page cannot do that safely, so with a private Space you test
with `curl` or the `/docs` page while logged in. Do not paste a token into the
demo page or commit one anywhere.

## 1. Create the Space

On huggingface.co: **New Space**, then:
- Name: for example `chess-detection`
- SDK: **Docker**, template **Blank**
- Hardware: **CPU basic (free)**
- Visibility: **Private**

## 2. Build the upload folder

From the repo root, on a checkout that has `models/synthetic.pt` and
`models/finetuned.pt`:

```bash
./scripts/prepare_space.sh
```

This writes `build/space/` (about 43 MB, gitignored) containing the
Dockerfile, `requirements-api.txt`, `src/`, the two weight files, and a Space
README whose header sets `sdk: docker` and `app_port: 8080`. The port matches
the `PORT=8080` the Dockerfile already uses. The container already runs as UID
1000, which Spaces require.

## 3. Upload

Log in with a token that has **write** permission (create it under Settings,
then Access Tokens). Run this yourself so the token stays out of this
session:

```bash
huggingface-cli login
huggingface-cli upload YOUR-USERNAME/chess-detection build/space . --repo-type=space
```

The CLI handles the large `.pt` files for you. Plain `git push` would need Git
LFS, which is not installed on this machine. If `huggingface-cli` complains,
check `huggingface-cli upload --help`, because the CLI has been renamed to `hf`
in newer versions.

## 4. Wait for the build

The Space page shows build logs. The first build installs PyTorch and takes a
few minutes. When it says Running, test it.

## 5. Test the private Space

Create a **read** token and keep it in an environment variable in your own
terminal:

```bash
export HF_TOKEN="hf_..."
```

The Space's direct URL is shown under **Embed this Space**, in the form
`https://USERNAME-SPACENAME.hf.space`:

```bash
export URL="https://YOUR-USERNAME-chess-detection.hf.space"
curl -H "Authorization: Bearer $HF_TOKEN" "$URL/health"
curl -H "Authorization: Bearer $HF_TOKEN" -X POST \
     "$URL/predict?model=finetuned" -F "file=@board.jpg;type=image/jpeg"
```

A 401 or 404 on a private Space usually means the token is wrong or lacks
read access. The first request after sleep is slow while the container starts.
Write down how long it takes; that is the real cold-start figure the README
still lists as TBD.

## Updating

Change the code, run `./scripts/prepare_space.sh` again, and re-run the upload
command. The Space rebuilds.

## Before making it public

1. Confirm the licence of the rendered dataset on its OSF page.
2. Decide what to do about the ChessReD weights (CC BY-NC-SA 4.0). Options are
   to ship only the synthetic model, or to ask the dataset authors.
3. Update `docs/MODEL_CARD.md` and the README credits to match.
4. Then change the Space to public in its settings and add the live link to
   the README.

## Not verified

- Creating the Space, the upload, the build on Hugging Face and the private
  token calls.
- Cold-start time and request latency on the free tier.
- Whether the free tier still has the hardware and sleep rules above.
- The 1 GB Space repo cap reported in a forum thread. The upload is 43 MB.
