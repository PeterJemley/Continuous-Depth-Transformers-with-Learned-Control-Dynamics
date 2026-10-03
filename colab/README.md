# Running jobs on Colab

Claude prepares each job in this repository. You run it on a Colab GPU with one fixed notebook, `colab/runner.ipynb`. The notebook downloads the latest code, runs whichever job `colab/NEXT_JOB.json` names, and uploads the results to the `colab-results` branch, where Claude reads them.

## One-time setup

### A. Create a GitHub token, so the notebook can upload results

1. On github.com, click your profile picture at the top right, then **Settings**.
2. At the bottom of the left sidebar, click **Developer settings**, then **Personal access tokens**, then **Fine-grained tokens**, then **Generate new token**.
3. Name it `colab-runner` and set **Expiration** to 30 days.
4. Under **Repository access**, choose **Only select repositories** and pick `Continuous-Depth-Transformers-with-Learned-Control-Dynamics`.
5. Under **Permissions**, open **Repository permissions** and set **Contents** to **Read and write**.
6. Click **Generate token** and copy it. GitHub shows it only once.

The token works only on this one repository and expires on its own. You can revoke it at any time on the same page.

### B. Open the notebook in Colab and keep a copy

1. Go to colab.research.google.com and choose **File > Open notebook**, then the **GitHub** tab.
2. Type `PeterJemley/Continuous-Depth-Transformers-with-Learned-Control-Dynamics`, choose the branch `claude/tender-goldberg-oojquh`, and click `colab/runner.ipynb`.
3. Choose **File > Save a copy in Drive**, and use that copy from now on. It always runs the newest code, because it downloads the code each time.

### C. Add the token as a Colab secret

1. In the notebook, click the **key icon** in the left sidebar. That opens **Secrets**.
2. Click **Add new secret**. Set the name to `GITHUB_TOKEN`, paste the token as the value, and turn on **Notebook access**.

### D. Choose a GPU

Choose **Runtime > Change runtime type**, select **T4 GPU**, and click **Save**.

## Each run

1. Open your saved copy of the notebook.
2. Choose **Runtime > Run all**.
3. Answer the prompts:
   - **Connect to Google Drive:** choose your account and allow access. Long jobs keep their checkpoints there, so they can resume after a disconnect.
   - **Secret access:** if Colab asks whether the notebook may use `GITHUB_TOKEN`, click **Grant access**.
   - **"This notebook was not authored by Google":** this appears only when opening from GitHub rather than your copy. Click **Run anyway**.
4. Keep the tab open until the last cell prints **Done**, then tell Claude.

Without a token, the last cell downloads a zip of the results instead. Attach it in the chat with Claude.

If Colab disconnects or runs out of GPU time, reopen the notebook and choose **Run all** again. Long jobs pick up from their last checkpoint on Drive.

Jobs work on the Colab machine's own disk and save copies to Drive in the background, so a slow or dropped Drive doesn't stop training. If Drive drops partway through, the session still finishes and uploads its results to GitHub. Only the checkpoints written after the drop are missing from Drive, which can mean a later session repeats some training.
