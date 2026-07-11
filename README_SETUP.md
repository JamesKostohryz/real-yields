# Real-Yield Tool — One-Time Setup

This sets up a free job that rebuilds the 30-year real-yield term structure
(nominal, real, breakeven, expected inflation + one-year forwards) every week
and feeds it into a Google Sheet automatically. After setup you do nothing —
the Sheet stays current on its own.

You'll do two short parts: **GitHub** (runs the job) and **Google Sheets**
(shows the data). Budget ~15 minutes. You never write code.

---

## Part A — GitHub (runs the weekly job)

**1. Make a free GitHub account** (skip if you have one): go to
https://github.com and sign up.

**2. Create the repository**
- Click the **+** (top right) → **New repository**.
- Repository name: `real-yields` (any name is fine).
- Set it to **Public**. *(Public is required so the Sheet can read the output,
  and it makes the job free. No secrets live in the repo — your FRED key is
  stored separately in step 4.)*
- Click **Create repository**.

**3. Upload the tool files**
- On the new repo page, click **Add file → Upload files**.
- Drag in **everything** from the unzipped `asfp_tool` folder: the `asfp`
  folder, `requirements.txt`, `README_SETUP.md`, and the `.github` folder.
- Scroll down, click **Commit changes**.
- ⚠️ If the `.github` folder didn't upload (some browsers hide dot-folders),
  create it by hand: **Add file → Create new file**, type
  `.github/workflows/weekly.yml` as the name, paste the contents of the
  `weekly.yml` file (also shown at the bottom of this guide), **Commit**.

**4. Add your FRED API key as a secret**
- In the repo: **Settings → Secrets and variables → Actions**.
- Click **New repository secret**.
- Name: `FRED_API_KEY`  — Value: paste your FRED key. Click **Add secret**.

**5. Run it once**
- Go to the **Actions** tab. If prompted, click to enable workflows.
- Click **weekly-real-yields** on the left → **Run workflow** → **Run workflow**.
- Wait 1–2 minutes and refresh. A green ✓ means success.
- ❌ If it's a red ✗, click into the run, open the failed step, copy the error
  text, and send it to me — first-run tweaks are normal.

**6. Confirm the output**
- Back on the repo main page, open the **outputs** folder. You should see
  `curve_latest.csv`, `headline_latest.csv`, and `history.csv`.

---

## Part B — Google Sheets (shows the data)

**7.** Open a new Google Sheet (sheets.new).

**8.** In cell **A1**, paste this — replacing `USERNAME` and `real-yields`
with your GitHub username and repo name:

```
=IMPORTDATA("https://raw.githubusercontent.com/USERNAME/real-yields/main/outputs/curve_latest.csv")
```

The full 1–30 curve fills in automatically.

**9.** (Optional) On a second tab, paste in A1 for the headline summary:

```
=IMPORTDATA("https://raw.githubusercontent.com/USERNAME/real-yields/main/outputs/headline_latest.csv")
```

That's it. The job reruns every Wednesday and the Sheet re-pulls the new
numbers on its own. To force a manual refresh any time, use **Run workflow**
in the Actions tab (step 5).

---

## What the columns mean

- **nominal / real / breakeven / exp_inflation** — the four term structures (%)
- **\*_fwd1y** — the one-year forward for that curve (e.g. `real_fwd1y` at
  maturity 10 is the 1-year real rate expected to prevail from year 9 to 10)
- **phi** — the premium (breakeven minus expected inflation)
- **provenance** — `observed` (real TIPS data) vs `front/back-constructed` (model)
- **reliability** — 1 = fully data-driven, lower = more model

---

## Notes

- The underlying Fed data updates **weekly** (Tuesdays), so the curve is "as of
  the most recent Friday." A weekly refresh is the right cadence.
- `history.csv` accumulates every run; because GitHub keeps every commit, you
  also get a full point-in-time archive automatically.
- Front-end (1–2y) and long-end (20–30y) points are model-constructed and
  carry more uncertainty than the 2–20y observed middle. This is a v1 with
  sensible default calibration; the front premium can be refined against
  history later.

---

## weekly.yml (only needed if the .github upload failed in step 3)

See the file `.github/workflows/weekly.yml` in this package — open it in a text
editor and copy its contents.
