"""Reproduce the paper's three-day carbon accounting example.

See README.md for the required Electricity Maps data.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CI_ZONE = "US-CAL-CISO"


def load_carbon_intensity(path):
    """Read exactly 72 hourly CAISO observations in the paper's local-time window."""
    if not path.is_file():
        raise ValueError(
            f"Carbon-intensity file not found: {path}. "
            "Acquire the Electricity Maps data as described in README.md, "
            "then pass --carbon-intensity PATH."
        )
    data = pd.read_csv(path)
    if not {"datetime", CI_ZONE}.issubset(data.columns):
        raise ValueError(f"Carbon-intensity CSV requires datetime and {CI_ZONE} columns.")
    times = pd.DatetimeIndex(pd.to_datetime(data["datetime"]))
    # Interpret naive timestamps as California local time.
    if times.tz is None:
        times = times.tz_localize("America/Los_Angeles")
    else:
        times = times.tz_convert("America/Los_Angeles")
    order = np.argsort(times)
    expected = pd.date_range("2024-05-10", periods=72, freq="h", tz="America/Los_Angeles")
    if not times[order].equals(expected):
        raise ValueError(
            "Carbon-intensity CSV must contain exactly 72 unique hourly observations "
            "from 2024-05-10 00:00 through 2024-05-12 23:00 America/Los_Angeles "
            "(2024-05-10 07:00Z through 2024-05-13 06:00Z), without gaps."
        )
    values = pd.to_numeric(data[CI_ZONE], errors="raise").to_numpy(dtype=float)[order]
    if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("Carbon intensity must be finite and nonnegative in gCO2eq/kWh, with a positive sum.")
    return values


def load_trace(path, scale):
    """Load and scale the three-day workload, counting context and generated tokens."""
    df = pd.read_csv(path)
    df["minute"] = pd.to_datetime(df["minute"], utc=True)
    expected_minutes = pd.date_range("2024-05-12", periods=4320, freq="min", tz="UTC")
    if not pd.DatetimeIndex(df["minute"]).equals(expected_minutes):
        raise ValueError("The bundled inference trace must contain 4,320 consecutive UTC minutes starting 2024-05-12.")
    counts = df[["context_tokens", "generated_tokens"]].to_numpy(dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts.sum(axis=1) <= 0).any():
        raise ValueError("Inference token counts must be finite, nonnegative, and nonzero in each minute.")
    t = df["minute"].dt.tz_localize(None)
    toks = counts.sum(axis=1) / scale
    hour_id = ((df["minute"] - df["minute"].iloc[0]).dt.total_seconds() // 3600).astype(int).values
    return t, toks, hour_id


def plot_example(t, hour_id, toks, ci_per_min, pool_load, components, reported, output_path):
    """Plot hourly intensities and their totals beside the provider report."""
    osci, m_per_tok, res_e, res_k = components

    def _hourly(arr):
        # Token-weighted hourly mean = the metric's reporting cadence (grid CI is hourly).
        _c = np.bincount(hour_id, weights=arr * toks)
        _k = np.bincount(hour_id, weights=toks)
        return (_c / _k)[hour_id]

    _op = _hourly(osci)
    _C_OP, _C_E, _C_K = "#3b6ba5", "#9ecae1", "#c44e52"  # operational / energy / capacity
    _C_TD2, _C_TD3 = "#b0b0b0", "#6b6b6b"                # top-down Scope-2 / Scope-3 greys

    def _kg(arr):
        return float((arr * toks).sum() / 1000.0)

    _op_kg, _emb_kg = _kg(osci), _kg(m_per_tok)
    _e_kg, _k_kg = _kg(res_e), _kg(res_k)
    _prov_kg = _op_kg + _e_kg + _k_kg              # provider top-down total (= rSCI)

    _fig = plt.figure(figsize=(6, 6))
    _gs = _fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.2)
    _gs_top = _gs[0].subgridspec(1, 2, width_ratios=[5, 1], wspace=0.05)
    _gs_bot = _gs[1].subgridspec(3, 2, width_ratios=[5, 1], wspace=0.05, hspace=0.12)
    _axes = [_fig.add_subplot(_gs_top[0, 0])]
    _axes += [_fig.add_subplot(_gs_bot[_r, 0], sharex=_axes[0]) for _r in range(3)]

    _l_ci, = _axes[0].plot(t, ci_per_min, color="black", alpha=0.8, lw=1)
    _axes[0].set_ylabel("carbon intensity\ngCO$_2$/kWh")
    _axes[0].set_ylim([0, 410])
    _axes[0].set_yticks([0, 200, 400])
    _ax0b = _axes[0].twinx()
    _l_load, = _ax0b.step(t, pool_load, where="post", color=_C_K, lw=1.2)
    _ax0b.set_yticks([0, 0.5, 1.0])
    _ax0b.set_ylabel("Pool load\n(normalized)")
    _ax0b.set_ylim([0, 410 / 400])
    _ax0b.legend([_l_load, _l_ci], ["pool load", "carbon intensity"], loc="upper left", frameon=False, bbox_to_anchor=(0.15, 1.3), ncol=2, columnspacing=1, handletextpad=0.5)
    _ax0b.tick_params(labelsize=9)
    _axes[0].set_title("Inputs", fontsize=10, weight="bold", loc="left")

    _axes[0].spines[["top"]].set_visible(False)
    _ax0b.spines[["top"]].set_visible(False)

    _axes[1].stackplot(t, _op * 1e6, colors=[_C_OP], labels=["operational"], alpha=0.95)
    _axes[2].stackplot(t, _op * 1e6, _hourly(m_per_tok) * 1e6,
                       colors=[_C_OP, _C_K], labels=["operational", "embodied"], alpha=0.95)
    _axes[1].set_ylabel("oSCI\ngCO$_2$/Mtok")
    _axes[1].set_title("Per-workload metrics", fontsize=10, weight="bold")
    _axes[2].set_ylabel("SCI\ngCO$_2$/Mtok")

    _axes[3].stackplot(
        t, _op * 1e6, _hourly(res_e) * 1e6, _hourly(res_k) * 1e6,
        labels=["operational", "energy-driven residual", "capacity-driven residual"],
        colors=[_C_OP, _C_E, _C_K], alpha=0.95)
    _axes[3].set_ylabel("rSCI\ngCO$_2$/Mtok")

    for _ax in _axes:
        _ax.grid(alpha=0.3)
    for _ax in _axes[:-1]:
        _ax.tick_params(labelbottom=False)  # dates only on the bottom row

    _axes[-1].set_xlim(t.min(), t.max())  # clip padding so only interior midnights tick
    _axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    _axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a"))
    for _lbl in _axes[-1].get_xticklabels():
        _lbl.set_rotation(0)
        _lbl.set_ha("center")

    _captured = {
        1: [(_op_kg, _C_OP)],                                    # oSCI: operational only
        2: [(_op_kg, _C_OP), (_emb_kg, _C_K)],                   # SCI: + embodied M
        3: [(_op_kg, _C_OP), (_e_kg, _C_E), (_k_kg, _C_K)],      # rSCI: full
    }
    _td = [(reported[0], _C_TD2, "S2"), (reported[1], _C_TD3, "S3")]  # reported scopes
    for _r in (1, 2, 3):
        _bax = _fig.add_subplot(_gs_bot[_r - 1, 1])
        if _r == 1:
            _bax.set_title("Total comparison", fontsize=10, weight="bold")
        _bottom = 0.0
        for _val, _col in _captured[_r]:
            _bax.bar(0, _val, bottom=_bottom, width=0.8, color=_col, edgecolor="white", linewidth=0.4)
            _bottom += _val
        _bottom = 0.0
        for _val, _col, _lab in _td:
            _bax.bar(1, _val, bottom=_bottom, width=0.8, color=_col)
            _bax.text(1, _bottom + _val / 2, _lab, ha="center", va="center", fontsize=9, weight='bold', color="white")
            _bottom += _val
        for _x, _total in [(0, sum(value for value, _ in _captured[_r])), (1, _prov_kg)]:
            _bax.text(_x, _total + 2, f"{_total:.1f}", ha="center", va="bottom", fontsize=8)
        _bax.spines[["top", "left"]].set_visible(False)
        _bax.yaxis.tick_right()
        _bax.yaxis.set_label_position("right")
        _bax.set_ylabel("Total carbon\nkg CO$_2$")
        _bax.set_xlim(-0.6, 1.6)
        _bax.set_ylim(0, max(_prov_kg, _kg(osci + m_per_tok)) * 1.20)
        _bax.set_xticks([0, 1] if _r == 3 else [])
        _bax.set_yticks([0, 50, 100])
        _bax.tick_params(labelsize=9)
    _fig.axes[-1].set_xticklabels(["Agg.", "Top\ndown"], fontsize=10)

    for _ax in _axes:
        _ax.tick_params(labelsize=9)

    for _ax in _axes[1:]:
        _ax.set_ylim(0, 600)
        _ax.spines[["top", "right"]].set_visible(False)

    _axes[1].legend(loc="upper left", frameon=False, ncols=1, bbox_to_anchor=(-0.01, 1), columnspacing=1, handletextpad=0.3)
    _axes[2].legend(loc="upper right", frameon=False, ncols=1, bbox_to_anchor=(1, 1), columnspacing=1, handletextpad=0.3, labelspacing=.2)
    _h, _l = _axes[3].get_legend_handles_labels()
    _order = [0,2,1]
    _axes[3].legend([_h[i] for i in _order],[_l[i] for i in _order], loc="upper left", frameon=False, ncols=2, bbox_to_anchor=(-0.01, 1.15), columnspacing=-5.2, handletextpad=0.3, labelspacing=.2)

    _fig.align_ylabels()
    _fig.savefig(output_path, bbox_inches="tight", metadata={"Creator": "Matplotlib", "Author": "", "Title": "Illustrative carbon accounting example"})
    plt.close(_fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carbon-intensity", type=Path, default=ROOT / "data/carbon_intensity.csv",
                        help="Electricity Maps CSV (see README.md)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output",
                        help="directory for the figure, metrics, and hourly allocations")
    args = parser.parse_args()
    # Illustrative assumptions for one 8xH100 node.
    TPS = 3000.0          # assumed aggregate throughput, prefill+decode (tokens/s)
    PEAK_W = 10200.0      # node power at full load (NVIDIA DGX H100 max)
    IDLE_W = 2500.0       # assumed node idle draw
    TE_KG = 2500.0        # assumed full-node embodied carbon
    LIFESPAN_YR = 4.0     # SCI-spec amortization lifespan
    PUE = 1.2             # facility overhead, folded into the bottom-up model

    S2_REPORTED = 80.0    # reported Scope 2 (kg)
    S3_REPORTED = 20.0    # reported Scope-3 (kg): embodied hardware/buildings + upstream electricity (FERA)
    # Assumed energy-driven fractions of the residuals.
    BETA_S2 = 0.10
    BETA_S3 = 0.50

    SCALE = 50.0          # scales the fleet trace to one node
    J_PER_KWH = 3_600_000.0

    GAMMA = (PEAK_W - IDLE_W) / TPS  # marginal energy per processed token (J/tok)

    t, toks, hour_id = load_trace(ROOT / "data/inference_trace.csv", SCALE)

    _ci_hourly = load_carbon_intensity(args.carbon_intensity)
    ci_per_min = _ci_hourly[hour_id]

    E_kwh = toks * GAMMA / J_PER_KWH                     # IT energy per minute (no PUE)
    osci = GAMMA * PUE * ci_per_min / J_PER_KWH          # gCO2/tok (varies with grid CI)
    print(f"total kWh={E_kwh.sum():,.1f}  oSCI(mean)={osci.mean():.6f} gCO2/tok  "
          f"op={(osci * toks).sum() / 1000:.1f} kg")

    _op_kg = (osci * toks).sum() / 1000.0
    RESIDUAL_S2 = S2_REPORTED - _op_kg   # idle draw oSCI omits + energy-model error
    RESIDUAL_S3 = S3_REPORTED            # embodied/buildings + FERA (bottom-up models none of S3)
    RESIDUAL_EN = BETA_S2 * RESIDUAL_S2 + BETA_S3 * RESIDUAL_S3
    RESIDUAL_CAP = (1 - BETA_S2) * RESIDUAL_S2 + (1 - BETA_S3) * RESIDUAL_S3
    print(f"op={_op_kg:.1f}  S2res={RESIDUAL_S2:.1f}  S3res={RESIDUAL_S3:.1f}  "
          f"EN={RESIDUAL_EN:.1f} CAP={RESIDUAL_CAP:.1f}  provider={_op_kg + RESIDUAL_S2 + RESIDUAL_S3:.1f}")

    embodied_g = np.full(len(toks), TE_KG * 1000.0 / (LIFESPAN_YR * 365.25 * 24 * 60))
    m_per_tok = embodied_g / toks
    sci = osci + m_per_tok

    # Synthetic shared-pool load, independent of the workload trace.
    pool_hours = np.arange(hour_id.max() + 1) + 0.5
    pool_load_hourly = 0.6 + 0.3 * np.cos(2 * np.pi * (pool_hours - 18.0) / 24.0)
    pool_load = pool_load_hourly[hour_id]

    # Illustrative pool-load-weighted temporal allocation, not a Shapley approximation.
    PEAK_P = 2.0

    _emissions = E_kwh * ci_per_min
    _wE = _emissions / _emissions.sum()
    res_e = _wE * RESIDUAL_EN * 1000.0 / toks  # gCO2/tok (tracks grid CI)

    _wK = pool_load**PEAK_P * toks
    _wK = _wK / _wK.sum()
    res_k = _wK * RESIDUAL_CAP * 1000.0 / toks  # gCO2/tok (rises with shared pool load)

    def _kg(arr):
        return float((arr * toks).sum() / 1000.0)

    # Independent hourly aggregation checks and reproducible manuscript values.
    hourly_tokens = np.bincount(hour_id, weights=toks)
    hourly_op = np.bincount(hour_id, weights=osci*toks)
    hourly_en = np.bincount(hour_id, weights=res_e*toks)
    hourly_cap = np.bincount(hour_id, weights=res_k*toks)
    assert np.isclose(_kg(osci + res_e + res_k), S2_REPORTED + S3_REPORTED)
    assert np.isclose(_kg(sci), _kg(osci) + embodied_g.sum()/1000)
    assert np.allclose(hourly_en / 1000, RESIDUAL_EN * hourly_op/hourly_op.sum())
    expected_capacity_weights = hourly_tokens * pool_load_hourly**PEAK_P
    expected_capacity_weights /= expected_capacity_weights.sum()
    assert np.allclose(hourly_cap / 1000, RESIDUAL_CAP * expected_capacity_weights)
    assert np.allclose(hourly_cap / hourly_tokens / pool_load_hourly**PEAK_P,
                       hourly_cap.sum() / (hourly_tokens * pool_load_hourly**PEAK_P).sum())
    assert np.all((pool_load_hourly > 0) & (pool_load_hourly <= 1))
    metrics = {
        "tokens": float(toks.sum()), "hours": int(len(hourly_tokens)),
        "operational_kg": _kg(osci),
        "embodied_kg": _kg(m_per_tok), "sci_kg": _kg(sci),
        "energy_residual_kg": RESIDUAL_EN, "capacity_residual_kg": RESIDUAL_CAP,
        "rsci_kg": _kg(osci + res_e + res_k),
        "uniform_capacity_g_per_mtok": RESIDUAL_CAP*1e9/toks.sum(),
        "pool_load_profile": "synthetic: 0.6 + 0.3*cos(2*pi*(h+0.5-18)/24), h=0,...,71",
        "pool_load_normalization": "aggregate demand / fixed available pool capacity",
        "capacity_weight_exponent": PEAK_P,
    }
    for name, arr in [("osci",osci),("sci",sci),("energy_residual",res_e),("capacity_residual",res_k)]:
        hourly = np.bincount(hour_id, weights=arr*toks)/hourly_tokens*1e6
        metrics[name+"_hourly_range_g_per_mtok"] = [float(hourly.min()),float(hourly.max())]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_example(
        t, hour_id, toks, ci_per_min, pool_load,
        (osci, m_per_tok, res_e, res_k), (S2_REPORTED, S3_REPORTED),
        args.output_dir / "sec5_example.pdf",
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    pd.DataFrame({"hour": np.arange(len(hourly_tokens)), "tokens": hourly_tokens,
                  "normalized_pool_load": pool_load_hourly,
                  "operational_kg": hourly_op/1000, "energy_residual_kg": hourly_en/1000,
                  "capacity_residual_kg": hourly_cap/1000}).to_csv(args.output_dir / "hourly_allocations.csv",index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
