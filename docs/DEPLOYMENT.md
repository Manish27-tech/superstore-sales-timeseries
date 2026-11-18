# Deployment

## Streamlit Community Cloud

1. Push the repository to GitHub under `Manish27-tech`.
2. Sign in at https://share.streamlit.io with the GitHub account.
3. Click **New app**, choose the repository `superstore-sales-timeseries`,
   branch `main`, main file path `dashboard/app.py`.
4. The build installs `requirements.txt` (streamlit, plotly, pandas, numpy,
   scikit-learn, matplotlib, joblib). No secrets are required.
5. The app reads only `data/dashboard/*`, which is committed, so it starts
   without downloading the raw extract.

## Local

```bash
python -m pip install -r requirements.txt
python run.py all            # rebuilds every artefact
python run.py dashboard     # opens the app on http://localhost:8501
python scripts/check_artifacts.py
```

## Continuous integration

`.github/workflows/ci.yml` runs ruff and the full test suite on every push and
pull request. The test suite takes about two minutes because it retrains the
forecast once.
