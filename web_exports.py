"""Four-column Origin output and compact scientific-parameter output.

Kept separate from the web GUI so the export behavior can be unit-tested.
"""

import numpy as np
import pandas as pd

from solar_jv_engine import internal_fit_to_raw_units, raw_y_label


def origin_four_column_table(V_raw, Y_raw, V_grid, J_grid, input_mode,
                             area_cm2, flip_voltage=False, flip_current=False):
    """The desktop application's unchanged Origin export semantics.

    All original numeric rows are preserved, in acquisition order; 600 smooth
    predicted points appear in independent columns (with blank padding).
    """
    V_out, Y_out = internal_fit_to_raw_units(
        V_grid, J_grid, input_mode, area_cm2,
        flip_voltage=flip_voltage, flip_current=flip_current,
    )
    raw_y_name, fit_y_name = raw_y_label(input_mode)
    return pd.DataFrame({
        "Voltage_raw_V": pd.Series(np.asarray(V_raw, dtype=float)),
        raw_y_name: pd.Series(np.asarray(Y_raw, dtype=float)),
        "Voltage_fit_V": pd.Series(np.asarray(V_out, dtype=float)),
        fit_y_name: pd.Series(np.asarray(Y_out, dtype=float)),
    })


def compact_parameter_table(fit):
    """Exactly 4 single-diode or 6 two-diode fitted parameters."""
    two = fit["model"] == "Two diode"
    keys = ("J01", "n1", "J02", "n2", "rs", "rsh") if two else (
        "J0", "n", "rs", "rsh",
    )
    return pd.DataFrame([{
        "Parameter": k,
        "Value": float(fit[k]),
        "Unit": "A/cm^2" if k.startswith("J0") else (
            "ohm cm^2" if k in ("rs", "rsh") else "dimensionless"
        ),
    } for k in keys])


def to_tsv(df):
    return df.to_csv(sep="\t", index=False, float_format="%.10e")
