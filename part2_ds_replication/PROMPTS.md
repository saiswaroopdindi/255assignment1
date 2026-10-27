# Prompts — Part 2 (replicating `dlmastery/data_science_examples`)

The original prompts come from the professor's [PROMPTS.md](https://github.com/dlmastery/data_science_examples/blob/main/PROMPTS.md).
They were written for a multi-agent `/teamwork-preview` workflow that builds full-stack FastAPI + React apps. I replicated
the **data-science core** of five projects with an agentic coding assistant (Claude in Cowork). Each project is a single
reproducible script/notebook, and one Streamlit admin dashboard covers all five.

## Original prompts (verbatim)

**01 · NYC Taxi**
```text
Now on to another project - You will do end2end a data science project of including data, training, deployment, crisp-dm framework, and an awesome front end. you can use Kaggle NYC taxi challenge. make sure that frontend includes interactive map and trip estimation.
```
Follow-ups: `make the admin dashboard even better with more data scientist related details. prepare a research report with crisp-dm methodology for the same as well.` · `please follow the autoresearch repo skills deligently for hill climbing. also match the dashboard details with research paper.`

**03 · Clustering**
```text
/teamwork-preview Now lets do another project - clustering using popular kaggle data set - make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement autoresearch to do hill climbing and match the dashboard details with research paper. include all details a data scientist and ai engineer will care.
```

**04 · Associative pattern mining**
```text
/teamwork-preview Now lets do another project - associative pattern mining using popular kaggle data set - make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement autoresearch to do hill climbing and match the dashboard details with research paper. include all details a data scientist and ai engineer will care.
```

**06 · Anomaly detection**
```text
/teamwork-preview Now lets do another project - anomaly detection using popular kaggle data set and popular methods - make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement autoresearch to do hill climbing and match the dashboard details with research paper. include all details a data scientist and ai engineer will care.
```

**12 · Time series**
```text
/teamwork-preview build a similar website in directory timeseries_test for timeseries forcasting similar to other projects you built - including detailed crisp-dm steps and admin and other dashboards
```

**Audit prompt (applied to all five)**
```text
use proepr data science skills to do very detailed deata science audit of code etc,. to ensur ethere is no leakage and reward hacking etc.,. and generate a proper report in per project basis
```

## Prompt I gave my coding assistant (adapted)

```text
Replicate the data-science experiments from github.com/dlmastery/data_science_examples projects 01, 03, 04, 06 and 12 using
the prompts in its PROMPTS.md. For each: follow CRISP-DM, use a popular public/Kaggle dataset that downloads without a
login, implement an "autoresearch" hill-climbing loop that logs every experiment (kept and rejected), use leakage-safe
temporal or out-of-fold validation, generate figures + metrics.json + an auto-written RESULTS.md with a plain-English
paraphrase, and build one Streamlit admin dashboard that shows all projects. Then have an independent reviewer audit
the code for leakage, API bugs and over-claimed results, and fix everything it finds.
```

## How the replication differs from the original
* **Scope:** data-science pipeline + Streamlit dashboard instead of FastAPI/React full-stack apps (same methods and metrics).
* **Datasets:** public mirrors that need no Kaggle login: NYC TLC open data, UCI Online Retail, OpenML credit-card fraud (same file as Kaggle), and UCI Bike Sharing.
* **Audit fixes applied:** temporal splits everywhere, out-of-fold target encoding, thresholds/hyper-parameters tuned only on validation data, test sets touched once.
