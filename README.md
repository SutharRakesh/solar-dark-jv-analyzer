# Solar-cell Dark J-V Analyzer (Streamlit web app)

An upload-and-fit web application built from the scientific functions of the supplied desktop `dark_jv_solar_updated.py` program. Uses the same single-diode and two-diode models, region suggestions, model fitting and diagnostics. No PySide6 or Qt required on the server.

## Web-app features

- Users upload a two-column `.txt`, `.dat`, `.tsv`, or `.csv` file or paste the data.
- Select units (mA/cm², A/cm², A), device area, polarity, temperature, model, and fit objective.
- Automatically suggested `Rsh`, `n/J0`, `Rs` initial estimate windows and final fit range, with manually adjustable sliders and shaded charts.
- Single or two diode equivalent-circuit fitting. Optional parameter locks, constrained `n1 = 1, n2 = 2` mode, and parameter-reliability heuristics.
- Linear current, log absolute current, and residual plots.
- Origin-ready **exactly four columns**: full measured V, full measured J/I, smooth fitted V, smooth fitted J/I in the same units as uploaded, preserving acquisition order.
- Compact parameters only: `J0, n, rs, rsh` or `J01, n1, J02, n2, rs, rsh`; area resistances use `ohm cm²`.
- Example data included so visitors can test without bringing a dataset.

## Run locally

Requires Python 3.10+ (recommend Python 3.12).

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local address printed in the terminal, usually `http://localhost:8501`.

## Publish as a PUBLIC app with a shareable URL

1. Create/sign in to a GitHub account: https://github.com/.
2. Create a new **public** repository, e.g. `solar-dark-jv-web`.
3. Upload the contents of this folder to the **root** of the repository. Include `streamlit_app.py`, `solar_jv_engine.py`, `web_exports.py`, `requirements.txt`, `example_solar_dark_jv_mAcm2.txt`, and the `.streamlit` directory. You do NOT need the older PySide6 file.
4. Sign into Streamlit Community Cloud: https://share.streamlit.io/.
5. Click **Create app** → **Yup, I have an app**. Choose the GitHub repository and branch `main`, set entrypoint to `streamlit_app.py`.
6. Choose a memorable available subdomain, e.g. `solar-dark-jv-analyzer`, and click **Deploy**.
7. Once it is running, click **Share** and ensure public access is enabled. Share the resulting `https://<chosen-name>.streamlit.app/` URL.

The deployed Streamlit link is the **public** link; the local `localhost` URL is only available to your own computer.

## Notes on science and privacy

- The code fits equivalent-circuit models. Even high R² does not establish that all fitted parameters are identifiable. Review warnings.
- Automatically suggested regions are heuristics; adjust them for device physics, current compliance, hysteresis, or non-ohmic leakage.
- The two-diode constrained idealities `n1 = 1`, `n2 = 2` are **optional** and not universally physical.
- Uploaded data are handled per browser session and not intentionally written to a persistent database by this source code. Do not assume the third-party hosting provider retains no request metadata. Do not upload confidential data to a public demo without assessing the hosting terms.
- Free hosting is appropriate for demonstration or moderate usage; heavy concurrent fits may require paid hosting or safeguards. This public-demo version rejects datasets above 5,000 numeric rows rather than silently dropping points.
