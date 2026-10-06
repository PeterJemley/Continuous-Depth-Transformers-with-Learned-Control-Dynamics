# Running jobs on Colab

Each job is prepared in this repository. You run it on a Colab GPU with one fixed notebook, `colab/runner.ipynb`. The notebook downloads the latest code, runs whichever job `colab/NEXT_JOB.json` names, and uploads the results to the `colab-results` branch, where they are analysed.

## One-time setup

### A. Create a GitHub token, so the notebook can upload results

1. On github.com, click your profile picture at the top right, then **Settings**.
2. At the bottom of the left sidebar, click **Developer settings**, then **Personal access tokens**, then **Fine-grained tokens**, then **Generate new token**.
3. Name it `colab-runner` and set **Expiration** to 30 days.
4. Under **Repository access**, choose **Only select repositories** and pick `Continuous-Depth-Transformers-with-Learned-Control-Dynamics`.
5. Under **Permissions**, click **+ Add permissions**, choose **Contents**, and set its access to **Read and write**. GitHub may add **Metadata** as read-only by itself; leave it. Leave the **Account** tab at 0.
6. Click **Generate token** and copy it. GitHub shows it only once.

The token works only on this one repository and expires on its own. You can revoke it at any time on the same page. The repository is private, so the notebook needs the token to download the code as well as to upload results.

### B. Upload the notebook to Colab

1. Get `colab/runner.ipynb`: either a copy sent to you directly, or from GitHub, by opening the file on the working branch and clicking the download button.
2. Go to colab.research.google.com and click **Upload notebook**, then choose the file. Colab saves it to your Google Drive.
3. From then on, open it from **Recents** on the Colab home page. It always runs the newest code, because it downloads the code each time.

The notebook itself changes rarely. When a new version is provided, upload it the same way and use it instead of the old one.

### C. Add the token as a Colab secret

1. In the notebook, click the **key icon** in the left sidebar. That opens **Secrets**.
2. Click **Add new secret**. Set the name to `GITHUB_TOKEN`, paste the token as the value, and turn on **Notebook access**.

After uploading a new version of the notebook, open the key icon again and check that `GITHUB_TOKEN` is listed, with **Notebook access** turned on for the new copy.

### D. Choose a GPU

Choose **Runtime > Change runtime type**, select **T4 GPU**, and click **Save**.

## Each run

1. Open your saved copy of the notebook.
2. Choose **Runtime > Run all**.
3. Answer the prompts:
   - **Connect to Google Drive:** choose your account and allow access. Long jobs keep their checkpoints there, so they can resume after a disconnect.
   - **Secret access:** if Colab asks whether the notebook may use `GITHUB_TOKEN`, click **Grant access**.
   - **"This notebook was not authored by Google":** this appears only when opening from GitHub rather than your copy. Click **Run anyway**.
4. Keep the tab open until the last cell prints **Done**, then report that it finished.

Without a token, the last cell downloads a zip of the results instead. Share it so the results can be analysed.

If Colab disconnects or runs out of GPU time, reopen the notebook and choose **Run all** again. Long jobs pick up from their last checkpoint on Drive.

Jobs work on the Colab machine's own disk and save copies to Drive in the background, so a slow or dropped Drive doesn't stop training. If Drive drops partway through, the session still finishes and uploads its results to GitHub. Only the checkpoints written after the drop are missing from Drive, which can mean a later session repeats some training.

## Tips

- **Run it from a computer that stays awake.** The training runs on Google's GPU, but free Colab ends a session if the browser tab goes idle or the computer sleeps. Nothing is lost, because the next **Run all** resumes, but an always-on computer means fewer interruptions.
- **Run one session at a time.** Two sessions running the same job would fight over the same files on Drive. Before starting on another computer, end the old session with **Runtime > Disconnect and delete runtime**.
- **"Automatic saving failed. This file was updated remotely or in another tab."** This is about saving the notebook's on-screen output, usually because the notebook is open in two tabs. It doesn't affect the job. Close any second tab and leave the running one alone.
- **"Cannot connect to GPU backend", or a usage-limit warning.** Free Colab limits daily GPU time. Wait a few hours, or until the next day, then choose **Run all** again.

