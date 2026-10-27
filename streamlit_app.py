"""Shareable solar-cell dark J-V analyzer using the existing fitting engine.

Run locally: streamlit run streamlit_app.py
Deploy: GitHub public repository -> Streamlit Community Cloud.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from solar_jv_engine import (
    SINGLE_KEYS, TWO_KEYS, PARAM_BOUNDS,
    _fit_core, estimate_all, parse_two_columns, raw_to_internal,
    solve_j_single, solve_j_two, suggest_regions, thermal_voltage,
)
from web_exports import compact_parameter_table, origin_four_column_table, to_tsv


st.set_page_config(
    page_title="Solar-cell Dark J–V Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2.3rem; max-width: 1900px;}
div[data-testid="stHorizontalBlock"] > div {min-width: 0;}
</style>
""", unsafe_allow_html=True)

REGIONS = ("rsh", "diode", "rs", "fit")
REGION_TITLES = {
    "rsh": "🔵 Rsh initial region",
    "diode": "🟢 n / J0 initial region",
    "rs": "🟠 Rs initial region",
    "fit": "⚫ Final nonlinear fit window",
}
REGION_COLORS = {
    "rsh": "rgba(55, 120, 245, 0.12)",
    "diode": "rgba(30, 150, 90, 0.12)",
    "rs": "rgba(245, 140, 30, 0.12)",
    "fit": "rgba(65, 65, 65, 0.07)",
}
OBJECTIVE_HELP = {
    "Balanced (asinh)": "Balances currents spanning many decades; works with positive and negative current. Good starting point if instrument noise is unknown.",
    "Linear J-V": "Minimizes absolute-current differences; high-current points often dominate.",
    "Log J-V (positive J only)": "Fits relative errors in the positive-current branch only. Reverse/zero-current points are NOT included.",
    "Noise-aware": "Divides current errors by a measurement uncertainty model. Supply an experimentally justified absolute noise floor.",
}


def digest(text, settings):
    return hashlib.sha256(text.encode("utf-8", "replace") +
                          json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()


def load_text(uploaded_file, pasted, demo):
    if demo:
        return Path(__file__).with_name("example_solar_dark_jv_mAcm2.txt").read_text(), "Synthetic example"
    if uploaded_file is not None:
        binary = uploaded_file.getvalue()
        if len(binary) > 5_000_000:
            raise ValueError("File is larger than 5 MB. Use a smaller two-column file.")
        return binary.decode("utf-8-sig", errors="replace"), uploaded_file.name
    if pasted.strip():
        return pasted, "Pasted measurements"
    return "", ""


def fallback_regions(V):
    vmin, vmax = float(np.min(V)), float(np.max(V))
    span = vmax - vmin
    return {
        "rsh": (vmin, vmin + 0.15 * span),
        "diode": (vmin + 0.4 * span, vmin + 0.7 * span),
        "rs": (vmin + 0.75 * span, vmin + 0.95 * span),
        "fit": (vmin, vmax),
    }


def make_figure(V, J, grid_v=None, grid_j=None, *, logarithmic=False,
                regions=None, show_shading=True):
    fig = go.Figure()
    if logarithmic:
        positive = J >= 0
        for selection, label, color in (
            (positive, "Measured J ≥ 0", "#1F77B4"),
            (~positive, "Measured J < 0 (absolute magnitude)", "#D55E00"),
        ):
            if np.any(selection):
                fig.add_trace(go.Scattergl(
                    x=V[selection], y=np.maximum(np.abs(J[selection]), 1e-30),
                    mode="markers", name=label, marker=dict(size=6, color=color),
                    hovertemplate="V=%{x:.5g} V<br>|J|=%{y:.5g} A/cm²<extra></extra>",
                ))
    else:
        fig.add_trace(go.Scattergl(
            x=V, y=J, mode="markers", name="Measured",
            marker=dict(size=6, color="#1F77B4"),
            hovertemplate="V=%{x:.5g} V<br>J=%{y:.5g} A/cm²<extra></extra>",
        ))
    if grid_v is not None and grid_j is not None:
        fit_y = np.maximum(np.abs(grid_j), 1e-30) if logarithmic else grid_j
        fig.add_trace(go.Scatter(
            x=grid_v, y=fit_y, mode="lines", name="Fitted model",
            line=dict(color="#D81B60", width=2.5),
        ))
    if show_shading and regions:
        for key in ("fit", "rsh", "diode", "rs"):
            lo, hi = regions[key]
            fig.add_vrect(x0=lo, x1=hi, fillcolor=REGION_COLORS[key],
                          line_width=0, layer="below")
    fig.update_layout(
        height=410, margin=dict(l=6, r=6, t=20, b=8),
        xaxis_title="Voltage (V)",
        yaxis_title="|J| (A/cm²)" if logarithmic else "J (A/cm²)",
        plot_bgcolor="white", legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="closest",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E9EDF2", zeroline=True)
    fig.update_yaxes(showgrid=True, gridcolor="#E9EDF2", zeroline=True,
                     type="log" if logarithmic else "linear")
    return fig


def sci(val):
    return f"{float(val):.4e}" if np.isfinite(val) else "—"


st.title("🔬 Solar-cell Dark J–V Analyzer")
st.caption("Upload or paste dark J–V data · Select physical regions · Fit a single- or two-diode model · Download Origin data and parameters")

left, middle, right = st.columns([1.20, 1.55, 3.30], gap="medium")

with left:
    st.subheader("1 · DATA")
    uploaded = st.file_uploader("Upload two-column data", type=["txt", "csv", "tsv", "dat"],
                                help="Column 1: voltage (V); column 2: J or I. Text header rows and # comments are permitted.")
    demo = st.checkbox("Use synthetic example", value=False,
                       help="Ideal single-diode example with known parameters for testing.")
    pasted = st.text_area("Or paste two columns", value="", height=125,
                          placeholder="-0.20  -0.004\n0.00  0.000\n0.30  0.010\n...",
                          help="If a file is uploaded, the file takes priority over pasted text.")
    input_mode = st.selectbox("Input units", [
        "V + J (mA/cm^2)", "V + J (A/cm^2)", "V + I (A)",
    ])
    area = st.number_input("Device area (cm²)", min_value=1e-10, max_value=1e6,
                           value=0.0256, format="%.8f", help="Used to convert measured current I (A) to J (A/cm²).")
    flip_v = st.checkbox("Flip voltage sign", value=False)
    flip_j = st.checkbox("Flip current sign", value=False)
    st.caption("The internal convention is positive forward voltage → positive dark current. All original numeric rows are retained.")

    try:
        raw_text, source_name = load_text(uploaded, pasted, demo)
    except ValueError as exc:
        st.error(str(exc))
        raw_text, source_name = "", ""

    V_raw = Y_raw = V = J = None
    data_signature = ""
    if raw_text:
        try:
            data = parse_two_columns(raw_text)
            vr = data["V_raw"].to_numpy(float)
            yr = data["Y_raw"].to_numpy(float)
            V, J, finite = raw_to_internal(
                vr, yr, input_mode, float(area),
                flip_voltage=flip_v, flip_current=flip_j,
            )
            V_raw, Y_raw = vr[finite], yr[finite]
            if len(V) < 8 or float(np.ptp(V)) <= 1e-10:
                raise ValueError("Use at least 8 numeric rows spanning a nonzero voltage range.")
            if len(V) > 5000:
                raise ValueError("The public web app supports up to 5,000 points per dataset. "
                                 "No points have been silently removed; use the desktop app "
                                 "for larger measurements.")
            data_signature = digest(raw_text, (input_mode, area, flip_v, flip_j))
            st.success(f"{len(V)} points loaded · {source_name}")
            st.caption(f"V: {np.min(V):.4g} to {np.max(V):.4g} V · J: {np.min(J):.3e} to {np.max(J):.3e} A/cm²")
            with st.expander("Preview uploaded data"):
                st.dataframe(data.head(25), hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read data: {exc}")
            V_raw = Y_raw = V = J = None
    else:
        st.info("Upload a .txt/.csv file, paste data, or select the synthetic example.")

with middle:
    st.subheader("2 · CONTROLS + REGIONS")
    T = st.number_input("Temperature (K)", min_value=1.0, max_value=1000.0,
                        value=300.0, step=1.0)
    model = st.selectbox("Diode model", ["Single diode", "Two diode"])
    objective = st.selectbox("Fitting objective", list(OBJECTIVE_HELP))
    st.caption(OBJECTIVE_HELP[objective])
    n_starts = st.slider("Multi-start attempts", min_value=1, max_value=30, value=8)
    check_sensitivity = st.checkbox("Check parameter reliability", value=True,
                                    help="Tests whether substantially different parameter values produce nearly the same fit; may take longer.")
    constrain_two = False
    if model == "Two diode":
        constrain_two = st.checkbox("Constrain n1=1, n2=2", value=False,
                                    help="Only use if these idealities are scientifically appropriate for your cell.")
    sigma_abs, sigma_rel = 0.0, 0.02
    if objective == "Noise-aware":
        sigma_abs = st.number_input("Measured absolute noise floor (A/cm²)",
                                     min_value=0.0, value=1e-10, format="%.3e")
        sigma_rel = st.number_input("Relative uncertainty (%)", min_value=0.0,
                                     max_value=100.0, value=2.0, step=0.5) / 100.0
        st.caption("Only use noise-aware fitting when these values have an experimental basis.")
    with st.expander("Advanced: fix selected parameters"):
        st.caption("Use only for values known independently. Otherwise let the program fit them.")
        fixed = {}
        names = TWO_KEYS if model == "Two diode" else SINGLE_KEYS
        defaults = dict(J0=1e-10, J01=7e-11, J02=3e-11,
                        n=1.5, n1=1.0, n2=2.0, rs=1.0, rsh=1e5)
        for key in names:
            if constrain_two and key in ("n1", "n2"):
                continue
            c1, c2 = st.columns([0.44, 0.56], gap="small")
            with c1:
                lock = st.checkbox(f"Fix {key}", key=f"lock_{key}")
            with c2:
                value = st.text_input(f"{key} value", value=str(defaults[key]),
                                      key=f"lockval_{key}", label_visibility="collapsed")
            if lock:
                try:
                    fixed[key] = float(value)
                except ValueError:
                    fixed[key] = float("nan")

    regions = None
    estimates = None
    region_notes = []
    if V is not None:
        suggest_key = digest(data_signature, {"T": T})
        try:
            suggested, region_notes = suggest_regions(V, J, float(T))
        except Exception as exc:
            suggested = fallback_regions(V)
            region_notes = [f"Automatic region suggestion failed: {exc}"]
        if st.session_state.get("region_source") != suggest_key:
            for key in REGIONS:
                st.session_state[f"region_{key}"] = tuple(suggested[key])
            st.session_state["region_source"] = suggest_key
        if st.button("AUTO SUGGEST REGIONS", use_container_width=True,
                     help="Restores physics-informed starting windows. Always inspect the shaded regions."):
            for key in REGIONS:
                st.session_state[f"region_{key}"] = tuple(suggested[key])
        vmin, vmax = float(np.min(V)), float(np.max(V))
        span = vmax-vmin
        step = max(1e-8, min(0.0001, span/1000.0))
        regions = {}
        for key in REGIONS:
            regions[key] = st.slider(
                REGION_TITLES[key], min_value=vmin, max_value=vmax,
                step=step, format="%.5f", key=f"region_{key}",
            )
        try:
            estimates = estimate_all(V, J, float(T), regions["rsh"],
                                     regions["diode"], regions["rs"])
            with st.expander("Initial estimates / region diagnostics"):
                st.write(f"J0: {sci(estimates.J0)} A/cm²")
                st.write(f"n: {estimates.n:.3g}" if np.isfinite(estimates.n) else "n: unavailable")
                st.write(f"rs: {sci(estimates.rs_area)} Ω·cm²")
                st.write(f"rsh: {sci(estimates.rsh_area)} Ω·cm²")
                for msg in (region_notes + estimates.warnings)[:7]:
                    st.caption("⚠ " + msg)
        except Exception as exc:
            st.error(f"Could not calculate initial estimates: {exc}")
        st.caption(f"Fit window contains {np.sum((V >= regions['fit'][0]) & (V <= regions['fit'][1]))} of {len(V)} points.")
    else:
        st.caption("Upload data to enable fitting regions.")

    run_fit = st.button("RUN NONLINEAR FIT", type="primary", use_container_width=True,
                        disabled=(V is None or regions is None or estimates is None))

# Store a fit keyed by EVERY setting affecting the result. Edited settings
# cannot silently display an old fitted curve or incorrect units.
fit_signature = ""
if V is not None and regions is not None:
    settings = dict(
        data=data_signature, regions={k: list(v) for k, v in regions.items()},
        T=T, model=model, objective=objective, n_starts=n_starts,
        constrain_two=constrain_two, sigma_abs=sigma_abs, sigma_rel=sigma_rel,
        fixed=fixed, sensitivity=check_sensitivity,
    )
    fit_signature = digest(data_signature, settings)

if run_fit and V is not None and estimates is not None and regions is not None:
    mask = np.isfinite(V) & np.isfinite(J) & (
        V >= regions["fit"][0]) & (V <= regions["fit"][1])
    vf, jf = V[mask], J[mask]
    try:
        if len(vf) < 8:
            raise ValueError("The fitting window needs at least 8 measured points.")
        if not all(np.isfinite(v) for v in fixed.values()):
            raise ValueError("An entered fixed parameter is not a valid number.")
        if objective == "Noise-aware" and sigma_abs <= 0:
            raise ValueError("Supply a positive experimentally measured noise floor (A/cm²).")
        if objective == "Log J-V (positive J only)" and int(np.sum(jf > 0)) < 4:
            raise ValueError("Log fit requires at least 4 positive-current points.")
        if np.sum(vf > 0) >= 5 and np.sum((vf > 0) & (jf > 0)) < 3:
            raise ValueError("Forward dark current appears negative. Check 'Flip current sign'.")
        with st.spinner("Fitting dark J–V and checking parameter reliability..."):
            result = _fit_core(
                vf, jf, float(T), estimates, objective, int(n_starts),
                two=(model == "Two diode"), fixed=fixed,
                constrain_two=constrain_two, sigma_abs=float(sigma_abs),
                sigma_rel=float(sigma_rel), check_sensitivity=check_sensitivity,
            )
            grid_v = np.linspace(float(np.min(vf)), float(np.max(vf)), 600)
            if result["model"] == "Single diode":
                grid_j = solve_j_single(grid_v, result["J0"], result["n"],
                                        result["rs"], result["rsh"], float(T))
            else:
                grid_j = solve_j_two(grid_v, result["J01"], result["n1"],
                                     result["J02"], result["n2"],
                                     result["rs"], result["rsh"], float(T))
        st.session_state["fit_record"] = {
            "signature": fit_signature, "result": result,
            "fit_v": vf, "fit_j": jf, "grid_v": grid_v, "grid_j": grid_j,
        }
    except Exception as exc:
        st.session_state.pop("fit_record", None)
        st.error(f"Fit failed: {exc}")

fit_record = st.session_state.get("fit_record")
if not fit_record or fit_record["signature"] != fit_signature:
    fit_record = None

with right:
    st.subheader("3 · PLOTS + PARAMETERS")
    shading = st.checkbox("Show fitting regions", value=True)
    tab_jv, tab_log, tab_res = st.tabs(["J–V", "Log |J|–V", "Residual"])
    if V is None:
        with tab_jv:
            st.info("Measured and fitted curves will appear here after data upload.")
        with tab_log:
            st.info("Upload data to inspect the logarithmic current-density curve.")
        with tab_res:
            st.info("Run a fit to see residuals.")
    else:
        grid_v = fit_record["grid_v"] if fit_record else None
        grid_j = fit_record["grid_j"] if fit_record else None
        with tab_jv:
            st.plotly_chart(make_figure(V, J, grid_v, grid_j,
                                        regions=regions, show_shading=shading),
                            use_container_width=True, config={"displaylogo": False})
        with tab_log:
            st.plotly_chart(make_figure(V, J, grid_v, grid_j,
                                        logarithmic=True, regions=regions,
                                        show_shading=shading),
                            use_container_width=True, config={"displaylogo": False})
            st.caption("Reverse-current values are displayed as |J|; blue/orange markers distinguish current signs.")
        with tab_res:
            if fit_record:
                ff = go.Figure()
                ff.add_trace(go.Scattergl(
                    x=fit_record["fit_v"], y=fit_record["result"]["raw_residual"],
                    mode="markers", name="Measured − predicted",
                    marker=dict(size=6, color="#1F77B4"),
                ))
                ff.add_hline(y=0, line_dash="dash", line_color="#666")
                ff.update_layout(height=390, margin=dict(l=5, r=5, t=10, b=8),
                                 xaxis_title="Voltage (V)",
                                 yaxis_title="Measured − fit (A/cm²)")
                st.plotly_chart(ff, use_container_width=True,
                                config={"displaylogo": False})
            else:
                st.info("Run the nonlinear fit to see measured − predicted current.")

    if fit_record:
        fit = fit_record["result"]
        st.success(f"{fit['model']} fit completed · {fit['fitted_count']} points")
        m1, m2, m3 = st.columns(3)
        m1.metric("R² (linear J)", f"{fit['R2']:.6f}")
        m2.metric("RMSE (A/cm²)", sci(fit["RMSE"]))
        m3.metric("Log RMSE (decades)", f"{fit['Log_RMSE']:.4g}" if np.isfinite(fit["Log_RMSE"]) else "—")
        st.caption("High R² alone does not prove the extracted diode parameters are unique or physically meaningful.")

        result_df = compact_parameter_table(fit)
        visible_df = result_df.copy()
        visible_df["Value"] = visible_df["Value"].apply(sci)
        st.dataframe(visible_df, hide_index=True, use_container_width=True)
        b1, b2 = st.columns(2)
        origin_df = origin_four_column_table(
            V_raw, Y_raw, fit_record["grid_v"], fit_record["grid_j"],
            input_mode, float(area), flip_v, flip_j,
        )
        with b1:
            st.download_button(
                "DOWNLOAD DATA FOR ORIGIN",
                data=to_tsv(origin_df), file_name="dark_jv_origin_4_columns.tsv",
                mime="text/tab-separated-values", use_container_width=True,
                help="4 columns: full raw V, full raw J/I, smooth fitted V, smooth fitted J/I in raw units.",
            )
        with b2:
            st.download_button(
                "DOWNLOAD PARAMETERS",
                data=to_tsv(result_df), file_name="dark_jv_parameters.tsv",
                mime="text/tab-separated-values", use_container_width=True,
                help="Exactly four parameters for the single-diode model or six for two diodes.",
            )
        if fit["warnings"]:
            with st.expander(f"⚠ Physics / parameter-reliability notes ({len(fit['warnings'])})", expanded=True):
                for warning in fit["warnings"]:
                    st.write("• " + warning)
        else:
            st.caption("No heuristic warnings. This is not proof of parameter uniqueness.")
    elif V is not None:
        if st.session_state.get("fit_record"):
            st.warning("Fitting settings or data changed. Click RUN NONLINEAR FIT for updated results.")
        else:
            st.info("Review the shaded regions, then click RUN NONLINEAR FIT.")

st.divider()
st.caption(
    "Research note: steady-state single/two-diode equivalent-circuit models are not universally valid "
    "at breakdown, contact-limited bias, current compliance, or for hysteretic sweeps. "
    "This app does not intentionally save uploads to a server-side database; "
    "review your hosting provider's data policy before uploading confidential measurements."
)
