# Methods

Short notes on what each part of `eegerds` computes, why, and where it stops being trustworthy.
Everything here is implemented in `src/eegerds/` and covered by tests; numbers quoted from
measurements come from the synthetic generator (`synth.py`), where the truth is known.

## 0. Conventions

* Signals are in **volts**. Continuous data are `(n_channels, n_samples)`; epochs are
  `(n_epochs, n_channels, n_times)` with a sampling rate `fs` and the time of the first sample
  `tmin` (seconds relative to the event).
* Epoch labels are MNE-style tags joined by `/` (`"left/high"`). `Epochs.select("left")` matches
  whole tags, so `"left"` never matches `"leftover"`.
* Every filter is zero phase (forward and backward), so there is no latency shift in ERPs and no
  timing shift in ERD/ERS.

## 1. Filters and their edges

Band-pass filters are Butterworth, built as second-order sections (`sosfiltfilt`); the notch is
`iirnotch` + `filtfilt` (Q = 30). A narrow band has a long memory, so a band-pass applied to each
epoch *separately* distorts the first and last half second. Measured on noise with a known flat
spectrum, band-passed 8-13 Hz and cut into 8 s epochs:

| way of filtering | power in the last 0.5 s | in the first 0.5 s |
|---|---|---|
| each epoch, SciPy's default padding | -18% | about -5% |
| each epoch, one second of odd-extension padding (what `erds` and `features` use) | -4% | smaller |
| the continuous recording, then epoching | 0 | 0 |

The last row is the recommended workflow for ERD/ERS: `bandpass_recording(rec, 8, 13)`, then
`epoch(...)`, then `ErdsConfig(prefiltered=True)`. **`prefiltered=True` on data that are not
limited to the band measures broadband power**, and the ERD comes out less than half as deep (in
the tests: about -20% instead of -48% for the simulated contralateral mu ERD). The default,
`prefiltered=False`, is the safe one.

## 2. ERD/ERS

```
ERD/ERS(t) = (P(t) - R) / R * 100          negative = ERD (decrease), positive = ERS
```

`P(t)` is the band power in a window around `t` (default 0.5 s, `overlap=0.5` means half a window,
so the stride is 0.25 s); `R` is the mean power over a reference interval (default -2 to -1 s).
Power is computed per sample from the band-passed epochs in one of two ways:

* `intertrial_variance` (default): the variance across epochs at each sample (`ddof=1`). It
  subtracts the across-epoch mean, i.e. the phase-locked (evoked) part, so only the *induced*
  activity is measured (Kalcher & Pfurtscheller, 1995).
* `power`: the mean of `x**2` across epochs (the classical method). It includes the evoked part, so
  a phase-locked burst inside the band shows up as ERS.

Things to know before reading a curve:

* **Smoothing is for display only** (`smoothed(n)`, a moving average over `n` windows with
  `mode="valid"`, so there is no edge bias). Overlap and smoothing make neighbouring values
  dependent; statistics should use unsmoothed, non-overlapping windows (`overlap=0`).
* **Every window is divided by the same `R`.** Its estimation error is therefore a common
  multiplicative error on the whole curve: about 12% in the ratio domain for a narrow band, 60
  epochs and a 1 s reference, and a measured bias of about +3 percentage points on unmodulated
  noise. More epochs, a longer reference and a wider band all reduce it.
* A zero reference power raises an error instead of returning `inf`.
* `trial_features` (section 8) works per epoch, in the log domain: `ln(P_window / P_reference)`.

## 3. Blink removal (SOBI)

SOBI (Belouchrani et al., 1997) separates sources that have different autocorrelations, which is
what a slow, large blink has and background EEG does not. `sobi` does:

1. centre the data and whiten with PCA (components with unit variance);
2. build symmetrised covariance matrices at lags `1 .. min(100, n/3)` of the whitened data;
3. jointly diagonalise them with real Jacobi rotations (the Cardoso-Souloumiac closed form:
   the rotation angle comes from the leading eigenvector of `sum g g^T`, with
   `g = (a_pp - a_qq, a_pq + a_qp)`), sweeping until the rotations are below `1/sqrt(n)/100`;
4. return mixing, unmixing and sources, sorted by the variance each source carries (`sum of mixing**2`).

It is deterministic (no random start). `remove_eog_components` scales the channels to unit SD,
correlates every source with a recorded EOG trace, and removes up to `n_remove` sources whose
`|r| >= min_abs_r` (default 0.3), reconstructing the rest. **A recording without blinks is left
untouched**: no source passes the correlation test, and the reconstruction is exact.

Limits: removing a component also removes whatever slow frontal background it shares with the
blinks; with one EOG trace only one kind of artefact can be targeted; the quality of the
separation depends on having enough samples (`n >= 10 * channels` is enforced).

## 4. Epoch rejection

For every channel a threshold is set from the data's own spread, `sd_factor * typical_sd`
(default 3), where `typical_sd` is the **median** per-epoch SD over all epochs (robust: it survives
up to half the epochs being contaminated) or, if you pass `reference` epochs known to be clean,
their mean. An epoch is rejected if any channel breaks any rule:

| rule | triggers when | default |
|---|---|---|
| `run` | more than this share of the epoch, *in a row*, is beyond `+-threshold` | 5% |
| `total` | more than this share of the epoch is beyond `+-threshold` (not in a row) | 10% |
| `amplitude` | any sample has `|x| > max_abs` | 150 uV |
| `variance` (optional) | any 0.5 s window has SD above `max_window_sd_ratio` x the channel's typical window SD | off |

Percentages become whole samples with `ceil(round(n * pct / 100, 9))` (so 5% of 200 samples is
exactly 10, not 10.000000000000002). The report says which rules fired for each epoch and which
channel did it.

The first three rules only see large, sustained deflections. On simulated contamination (bursts of
muscle-like noise, slow sway, electrode pops in 25% of the trials; 8 subjects x 40 trials, three
seeds) they caught 62-70% of the contaminated epochs; adding `max_window_sd_ratio=3` raised that
to 97-100%, with no clean epoch rejected in either setting. A real recording has contamination
that this generator does not, so check the report instead of trusting the rate.

## 5. ERP peaks

The components are searched **in order**, each in its own window and after the one it follows:

| component | polarity | window (s) | after | gap to the previous peak (s) |
|---|---|---|---|---|
| N200 | negative | 0.12 - 0.30 | | |
| P300 | positive | 0.20 - 0.55 | N200 | 0.02 - 0.27 |
| N400 | negative | 0.30 - 0.70 | P300 | 0.02 - 0.20 |

`PeakSpec` lets you change any of it. A component that cannot be searched (its window is empty, has
NaN, or the component it follows was not found) is returned with `found=False` and NaN; an extremum
sitting on the edge of its window sets `at_edge=True`, meaning the real peak may be outside.

Two choices that matter for inference:

* **Group first.** `group_erp` averages each subject's epochs and then the subjects, so a subject
  with more clean epochs does not weigh more.
* **Latency from the pooled average, amplitude at that latency.** The maximum of a noisy
  single-subject waveform is positive even in pure noise (in the tests, about +1.5 SD in a window
  where the true value is 0), so picking each subject's own peak inflates amplitudes. The group
  latency is read from the average of *all* conditions, which is blind to the contrast between
  them, and the amplitude of every subject and condition is the mean in `+-25 ms` around it.

## 6. Statistics across subjects

The unit of analysis is the subject. Each test compares one value per subject under two
conditions (or channels), for every window or component.

* **Wilcoxon signed-rank** (two-sided, SciPy). The effect size is the matched-pairs rank-biserial
  correlation `(W+ - W-) / (W+ + W-)`, between -1 and 1 (positive: first condition larger).
  Pairs with a NaN are dropped from that test only.
* **`p_floor`.** The smallest p an exact two-sided test can give with `n` pairs is `2 / 2**n`:
  0.0078 for 8 subjects, 0.031 for 6. It is a property of the test, not of the effect.
  Bonferroni over more than 6 windows can never reject anything with 8 subjects, and BH needs
  several windows sitting at the floor together.
* **Benjamini-Hochberg FDR** (`wilcoxon_fdr`, the default) over the whole table of one call.
* **Sign-flip max-T** (`signflip_maxt`, Nichols & Holmes, 2002): under the null the paired
  differences are symmetric around 0, so flipping the sign of a whole subject (all windows
  together) is a valid relabelling. For each relabelling the largest `|t|` over all windows is
  recorded; the adjusted p of a window is the share of relabellings whose maximum reaches its
  `|t|`. It controls the **family-wise** error rate, respects the correlation between neighbouring
  windows, and is *exact* up to 12 subjects (all `2**n` patterns are enumerated; its floor is
  also `2 / 2**n`, whatever the number of windows). Beyond that it draws `n_perm` random patterns
  and reports `(1 + count) / (1 + n_perm)`. Zero spread is decided relative to the size of the
  data, so identical differences give `t = +-inf` and `p = 2 / 2**n`, never 0.

* **Cluster-based permutation** (`cluster`, Maris & Oostenveld, 2007; `compare_conditions` and
  `compare_channels` only, because it needs neighbouring windows: `compare_amplitudes` refuses
  it). Within one series, neighbouring windows whose paired `|t|` is beyond the two-sided
  `cluster_p` threshold of the t distribution (default 0.05, `n - 1` degrees of freedom) and have
  the same sign form a cluster; its *mass* is the sum of `|t|` over its windows. A positive run
  next to a negative one is two clusters. The null distribution is built as for max-T (flip whole
  subjects, all series and windows together), but what is recorded for each relabelling is the
  largest cluster mass over all series, and the adjusted p of a cluster is the share of
  relabellings whose largest cluster reaches its mass. Every window of a cluster gets that p,
  windows outside any cluster get 1, and `significant` marks the windows of the clusters below
  `alpha`. The table gets two columns, `cluster` (1, 2, ... in series and time order; 0 outside)
  and `cluster_mass`; `p_raw` is NaN, because a window alone has no p, and `effect` is still the
  rank-biserial correlation of the window. The enumeration, the `2 / 2**n` floor and the
  random-pattern fallback are those of max-T. A window whose differences are all identical has
  `t = +-inf`; it is capped at 1e9 so that sums stay finite, and it dominates its cluster.
  Data with NaN are refused, as for max-T.

  What it says, with the family-wise error controlled over every series and window of the call:
  the conditions differ *somewhere*, and the cluster is where the evidence was. What it does not
  say: that each window of the cluster differs, or that the effect starts at its first window and
  ends at its last (Sassenhagen & Draschkow, 2019, Psychophysiology 56, e13335). `cluster_p` decides what may form
  a cluster, not what is significant; choose it before looking. With 8 subjects the exact test
  has 256 patterns, so the smallest p is 0.0078.

  The trade-off, measured (8 subjects, 16 windows, noise correlated between neighbouring windows
  with `r = 0.7`; the settings and seeds of `tests/test_cluster.py`): with no effect, at least one
  significant window in 4.5% of 400 experiments for the cluster test, 6.5% for max-T, 40% without
  correction and 0.25% for BH. With a weak effect spread over six windows (peak 1.2 standard
  deviations; 150 experiments), 69% for the cluster test, 45% for max-T, 17% for BH. With a
  2-standard-deviation effect in one window, 28% for the cluster test, 85% for max-T, 1% for BH.
  A sum of `|t|` over a run rewards width and a sharp single peak adds up to little, while the
  null distribution of the largest cluster mass holds the wider clusters that noise makes.

In a null experiment (8 subjects, 16 windows at C3, left and right trials identical; see
`examples/false_positive_windows.py`) the share of experiments with at least one significant
window (for the cluster test, at least one significant cluster) is shown in the README figure and
saved per experiment in `fwer_runs.csv`: 80 of 200 without correction, 0 for BH, 14 for max-T and
13 for the cluster test.

## 7. Synthetic generator

`simulate_recording` builds a continuous recording with a known truth: 12 channels with 2-D scalp
coordinates; mu (10.5 Hz) and beta (21 Hz) narrow-band noise multiplied by a slow envelope
`sqrt(1 + g(t))`, with `g` the exact ERD/ERS profile (contralateral ERD, a weaker ipsilateral ERD
set by `ipsi_ratio`, a beta rebound, an occipital rhythm that does not react); an N200/P300/N400
ERP (P300 larger in the "high" conditions); Poisson blinks projected onto frontal channels and a
recorded EOG trace; 1/f background, mains noise; and artefacts (bursts, sway, pops) in a fraction
of the trials. Every subject draws its own ERD depth, ERP amplitude and latency shift.

`expected_erds` gives the exact expected curve for any channel and band, and `truth` returns the
draws, blink signal and the artefact trials, which is what the tests and the demo compare against.
It is a model of what the *methods* do, not of the brain.

## 8. Features and the CSV contract

`trial_features` returns one row per epoch: `subject`, `level` (the class label), `trial`, and one
column per `<channel>_<band>_<start>-<stop>s` holding `ln(P_window / P_reference)` of that single
epoch. The same layout (`subject, level, ...` and numeric features, never any of the reserved
names `subject, session, trial, segment_id, kind, t_start, t_stop, label, level, x` as features) is
read by `subjectwise`, whose `evaluate` gives a leave-one-subject-out accuracy instead of the
optimistic one from a random split of epochs.

## 9. What is not here

No source localisation, no connectivity, no time-frequency decomposition beyond the band powers
above, no automatic bad-channel detection or interpolation, no re-referencing, and nothing
validated on real recordings by this package: the claims above are about the code, on synthetic
data. For those other things, use MNE-Python; `to_mne` and `from_mne` convert in both directions.
