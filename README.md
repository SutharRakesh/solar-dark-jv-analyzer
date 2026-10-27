# Solar Dark J–V Analyzer

### Scientific Analysis of Solar-Cell Dark Current–Voltage Characteristics

A Python-based scientific application for analyzing dark current–voltage (J–V) measurements of photovoltaic devices using single-diode and two-diode equivalent-circuit models.

The application provides an interactive Streamlit interface for nonlinear parameter extraction, visualization, and assessment of fitting quality.

## Scientific Capabilities

* Single-diode and two-diode equivalent-circuit modeling.
* Nonlinear fitting of experimental dark J–V characteristics.
* Extraction of saturation current density, diode ideality factors, series resistance, and shunt resistance.
* Automatic suggestions and manual adjustment of fitting regions.
* Multiple fitting objectives for measurements spanning different current ranges.
* Interactive linear-scale, logarithmic-scale, and residual plots.
* Export of measured and fitted data for further scientific analysis.
* Demonstration using an included synthetic dataset.

## Extracted Parameters

**Single-diode model**

\(J_0\) — Reverse saturation current density
\(n\) — Diode ideality factor
\(R_s\) — Series resistance
\(R_{sh}\) — Shunt resistance

**Two-diode model**

\(J_{01}, J_{02}\) — Saturation current densities
\(n_1, n_2\) — Diode ideality factors
\(R_s\) — Series resistance
\(R_{sh}\) — Shunt resistance

Area-normalized resistances are expressed in Ω·cm².

## Scientific Considerations

The application fits equivalent-circuit models to measured dark J–V characteristics.

Extracted parameters depend on experimental quality, the selected voltage range, measurement temperature, and optimization settings.

A good numerical fit does not necessarily imply that all parameters are uniquely identifiable or that individual recombination mechanisms have been established.

Experimental validation and careful interpretation remain essential.

## Running the Application

Python 3.10 or later is required.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the application:

```bash
python -m streamlit run streamlit_app.py
```

Access the application at:

`http://localhost:8501`

## Technology

**Programming language:** Python

**Web framework:** Streamlit

**Scientific computing:** NumPy and SciPy

**Data processing:** Pandas

**Visualization:** Plotly

## Repository

GitHub: `SutharRakesh/solar-dark-jv-analyzer`

## Project Scope

This software is intended for research-oriented analysis, visualization, and demonstration of dark J–V fitting methods for photovoltaic devices.
