"""Simulate a small group, clean it, and ask the usual questions.

    python examples/quickstart.py

Everything is synthetic: ``simulate_group`` draws subjects with a known ERD/ERS and ERP, so every
answer below can be compared with the truth (``eegerds demo`` does that and writes a report).
"""

import eegerds as ee


def main() -> None:
    # 1. Eight synthetic subjects: continuous 12-channel EEG with blinks, mains noise, artefacts.
    recordings = ee.simulate_group(8, ee.SimConfig(n_trials=10), seed=0)

    # 2. Notch + band-pass, SOBI blink removal, epoching, epoch rejection (with a reason per drop).
    config = ee.PipelineConfig(rejection=ee.RejectionConfig(max_window_sd_ratio=3.0))
    results = ee.process_group(recordings, config)
    first = results[0]
    assert first.report is not None and first.eog is not None
    print(f"{first.subject}: kept {first.epochs.n_epochs} of {first.epochs_all.n_epochs} epochs;")
    print(f"  rejected by rule: {first.report.counts()}")
    top = abs(first.eog.correlations[first.eog.removed[0]])
    print(f"  blink component removed (|r| with the EOG trace = {top:.2f})")
    epochs = [r.epochs for r in results]

    # 3. ERD/ERS in the mu band, left vs right hand, window by window (non-overlapping for tests).
    erds = ee.ErdsConfig(band=(8, 13), window=0.5, overlap=0.0)
    left = ee.erds_group([e.select("left") for e in epochs], erds)
    right = ee.erds_group([e.select("right") for e in epochs], erds)
    table = ee.compare_conditions(left, right, ["C3", "C4"])  # Wilcoxon, FDR over all windows
    best = table.nsmallest(4, "p_adj")[["series", "t_start", "t_stop", "mean_a", "mean_b", "p_adj"]]
    print("\nleft minus right, mu band (percent of reference power):")
    print(best.round(3).to_string(index=False))
    print(
        f"significant windows: {int(table['significant'].sum())} of {len(table)}; with "
        f"{left.n_subjects} subjects no p-value can be smaller than {table['p_floor'].iloc[0]:.4f}"
    )

    # 4. ERP components of the pooled average, then high vs low amplitudes at those latencies.
    pooled = ee.group_erp(epochs, baseline=(-0.2, 0.0))
    peaks = ee.find_group_peaks(pooled, ["Cz"])
    print("\npooled ERP at Cz:")
    for peak in peaks["Cz"]:
        print(f"  {peak.name}: {peak.latency * 1000:.0f} ms, {peak.amplitude * 1e6:+.1f} uV")
    high = ee.measure_amplitudes(ee.group_erp(epochs, ["high"], baseline=(-0.2, 0.0)), peaks)
    low = ee.measure_amplitudes(ee.group_erp(epochs, ["low"], baseline=(-0.2, 0.0)), peaks)
    print(ee.compare_amplitudes(high, low)[["component", "mean_a", "mean_b", "p_adj"]].round(7))

    # 5. Trial-level features for a classifier: subject, level, trial, then one column per feature.
    features = ee.features_table(
        [e.pick(["C3", "Cz", "C4"]) for e in epochs], classes=("left", "right")
    )
    print(
        f"\nfeature table: {len(features)} rows x {features.shape[1] - 3} features; first columns:"
    )
    print(features.iloc[:3, :6].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
