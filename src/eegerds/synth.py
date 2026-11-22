"""A synthetic EEG generator whose ground truth is known.

Why: every claim about a pipeline (this ERD is real, that component is a blink, the P300 is
larger here) needs data where the answer is known. This module simulates a continuous recording
of 12 channels with

* **sensorimotor rhythms** (mu ~10.5 Hz, beta ~21 Hz) whose power drops during movement
  preparation (ERD) and, in beta, rebounds afterwards (ERS), stronger over the hemisphere
  *contralateral* to the side of the trial;
* a **control rhythm** over the occipital cortex that does not react to the task;
* a **phase-locked ERP** (N200, P300, N400) whose P300 is larger in ``high`` conditions;
* **eye blinks** projected mainly onto frontal channels, plus a recorded EOG channel;
* occasional **artefacts** in a fraction of the trials: a muscle burst, a slow head-sway wave or
  an electrode pop, so that epoch rejection has something to find;
* 1/f background noise and a small line-noise sinusoid;
* between-subject variability in depth, ERP size and latency.

The rhythms are narrow-band noise multiplied by a slow envelope, so their phase is random from
trial to trial: the ERD is *not* phase-locked, as in real data. ``expected_erds`` gives the exact
expected ERD/ERS curve (in percent) for a channel, which the tests compare with what the
pipeline recovers.

Nothing here is a model of any real participant; the numbers are generic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .data import Recording
from .filters import bandpass

__all__ = [
    "DEFAULT_CHANNELS",
    "BANDS",
    "SimConfig",
    "expected_erds",
    "simulate_group",
    "simulate_recording",
]

#: 2-D scalp positions (x: left -> right, y: back -> front), arbitrary units.
DEFAULT_CHANNELS: dict[str, tuple[float, float]] = {
    "Fp1": (-0.30, 1.00),
    "Fp2": (0.30, 1.00),
    "F3": (-0.45, 0.55),
    "Fz": (0.00, 0.55),
    "F4": (0.45, 0.55),
    "C3": (-0.50, 0.00),
    "Cz": (0.00, 0.00),
    "C4": (0.50, 0.00),
    "P3": (-0.45, -0.55),
    "Pz": (0.00, -0.55),
    "P4": (0.45, -0.55),
    "Oz": (0.00, -1.00),
}

#: Analysis bands the expected curves are defined for.
BANDS: dict[str, tuple[float, float]] = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}

_PINK_LOW_HZ = 0.5
_SPATIAL_SIGMA = 0.30  # spread of the sensorimotor sources
_OCCIPITAL_SIGMA = 0.50
_NOISE_BANDWIDTH = {"mu": 3.0, "beta": 6.0}  # width of the simulated rhythm around its peak


@dataclass(frozen=True)
class SimConfig:
    """Parameters of the simulation. Every default is a plausible, generic value."""

    fs: float = 250.0
    n_trials: int = 10  # per condition
    conditions: tuple[str, ...] = ("left/low", "left/high", "right/low", "right/high")
    epoch: tuple[float, float] = (-2.0, 6.0)  # seconds around the cue; trials tile the recording
    erd: tuple[float, float] = (0.5, 3.0)  # ERD window, seconds after the cue
    ers: tuple[float, float] = (3.0, 4.5)  # rebound window
    mu_hz: float = 10.5
    beta_hz: float = 21.0
    mu_depth: float = 0.6  # fraction of the rhythm's power lost over the contralateral cortex
    beta_erd_ratio: float = 0.5  # beta ERD depth relative to the mu depth
    beta_rebound: float = 1.0  # relative power gain of the beta rebound (contralateral)
    ipsi_ratio: float = 0.25  # ipsilateral effects as a fraction of the contralateral ones
    rhythm_snr: float = 4.0  # rhythm power / background power inside its band, at its source
    background_uv: float = 8.0  # RMS of the 1/f background of each channel
    erp_uv: tuple[float, float, float] = (-4.0, 6.0, -3.0)  # N200, P300, N400 amplitudes
    erp_latency: tuple[float, float, float] = (0.20, 0.32, 0.46)
    erp_width: tuple[float, float, float] = (0.03, 0.05, 0.06)
    p300_gain_high: float = 1.6  # P300 multiplier in conditions tagged "high"
    blink_rate: float = 0.3  # blinks per second (0 disables them)
    blink_uv: float = 100.0  # amplitude of a blink at the source
    artifact_rate: float = 0.1  # fraction of trials hit by one artefact (0 disables them)
    artifact_uv: float = 60.0  # RMS of a muscle burst
    line_hz: float = 50.0
    line_uv: float = 3.0
    subject_sd: float = 0.15  # between-subject spread (log scale) of depth and ERP size

    def __post_init__(self) -> None:
        if self.fs < 100:
            raise ValueError("fs must be at least 100 Hz")
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least 1")
        if not self.epoch[0] < 0 < self.epoch[1]:
            raise ValueError("epoch must straddle the cue (tmin < 0 < tmax)")
        if not 0 <= self.mu_depth < 1:
            raise ValueError("mu_depth must be in [0, 1)")
        for cond in self.conditions:
            if not {"left", "right"} & set(cond.split("/")):
                raise ValueError(f"condition {cond!r} needs a 'left' or 'right' tag")
        if self.erd[1] > self.epoch[1] or self.ers[1] > self.epoch[1]:
            raise ValueError("ERD/ERS windows must end inside the epoch")
        if not 0 <= self.artifact_rate <= 1:
            raise ValueError("artifact_rate must be in [0, 1]")
        if self.span < 4.0:
            raise ValueError("epochs shorter than 4 s leave no room for the artefacts")

    @property
    def span(self) -> float:
        return self.epoch[1] - self.epoch[0]


# ---------------------------------------------------------------------------------------
# Helpers shared by the generator and by the analytic ground truth
# ---------------------------------------------------------------------------------------
def _coords(ch_names: Sequence[str]) -> np.ndarray:
    return np.array([DEFAULT_CHANNELS[c] for c in ch_names], dtype=float)


def _spatial_weights(coords: np.ndarray, centre: tuple[float, float], sigma: float) -> np.ndarray:
    d2 = ((coords - np.asarray(centre)) ** 2).sum(axis=1)
    return np.exp(-d2 / (2 * sigma**2))


def _ramp(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(x, 0.0, 1.0)))


def _box(t: np.ndarray, start: float, stop: float, edge: float = 0.3) -> np.ndarray:
    """Smooth box: raised-cosine edges of width ``edge`` centred on ``start`` and ``stop``."""
    return _ramp((t - start) / edge + 0.5) * _ramp((stop - t) / edge + 0.5)


def _side(condition: str) -> str:
    tags = condition.split("/")
    return "left" if "left" in tags else "right"


def _band_share(band: tuple[float, float], fs: float) -> float:
    """Share of the 1/f background variance that falls inside ``band``."""
    return math.log(band[1] / band[0]) / math.log((fs / 2) / _PINK_LOW_HZ)


def _gain(cfg: SimConfig, rhythm: str, hemisphere: str, side: str, t: np.ndarray, scale: float):
    """Relative power change g(t) of one rhythm source (power = baseline * (1 + g))."""
    contralateral = (side == "left" and hemisphere == "R") or (
        side == "right" and hemisphere == "L"
    )
    factor = 1.0 if contralateral else cfg.ipsi_ratio
    erd = _box(t, *cfg.erd)
    if rhythm == "mu":
        return -cfg.mu_depth * scale * factor * erd
    rebound = _box(t, *cfg.ers)
    return -cfg.beta_erd_ratio * cfg.mu_depth * scale * factor * erd + (
        cfg.beta_rebound * scale * factor * rebound
    )


def expected_erds(
    cfg: SimConfig,
    side: str,
    channel: str,
    band: str,
    times: np.ndarray,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """Expected instantaneous ERD/ERS (%) of ``channel`` in ``"mu"`` or ``"beta"``.

    ``side`` is ``"left"`` or ``"right"`` (the moving side); ``depth_scale`` is the subject's
    multiplier (``recording.truth["depth_scale"]``). The curve is the exact expectation of the
    ratio the pipeline estimates, assuming the rhythm lies inside the analysis band and the
    background is the simulated 1/f noise.
    """
    if band not in BANDS:
        raise ValueError(f"band must be one of {sorted(BANDS)}")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    names = list(DEFAULT_CHANNELS)
    if channel not in names:
        raise KeyError(f"unknown channel {channel!r}")
    coords = _coords(names)
    j = names.index(channel)
    times = np.asarray(times, dtype=float)
    bg_var = cfg.background_uv**2 * _band_share(BANDS[band], cfg.fs)
    src_var = cfg.rhythm_snr * bg_var
    centres = {"L": DEFAULT_CHANNELS["C3"], "R": DEFAULT_CHANNELS["C4"]}
    numerator = np.zeros_like(times)
    denominator = bg_var
    for hemi, centre in centres.items():
        w = _spatial_weights(coords, centre, _SPATIAL_SIGMA)[j]
        numerator += w**2 * src_var * _gain(cfg, band, hemi, side, times, depth_scale)
        denominator += w**2 * src_var
    if band == "mu":  # the occipital rhythm is also a mu-band source, and it never reacts
        w_occ = _spatial_weights(coords, DEFAULT_CHANNELS["Oz"], _OCCIPITAL_SIGMA)[j]
        denominator += w_occ**2 * src_var
    return 100.0 * numerator / denominator


def _pink_noise(rng: np.random.Generator, shape: tuple[int, int], fs: float) -> np.ndarray:
    n_ch, n = shape
    spec = np.fft.rfft(rng.standard_normal((n_ch, n)), axis=1)
    f = np.fft.rfftfreq(n, 1 / fs)
    shaping = np.zeros_like(f)
    shaping[f >= _PINK_LOW_HZ] = f[f >= _PINK_LOW_HZ] ** -0.5
    x = np.fft.irfft(spec * shaping, n=n, axis=1)
    return x / x.std(axis=1, keepdims=True)


def _narrowband(rng: np.random.Generator, n: int, fs: float, centre: float, width: float):
    x = bandpass(rng.standard_normal(n), fs, centre - width / 2, centre + width / 2, order=4)
    return x / x.std()


# ---------------------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------------------
def simulate_recording(
    config: SimConfig | None = None,
    seed: int | np.random.SeedSequence | None = 0,
    subject: str = "S01",
) -> Recording:
    """Simulate one subject's continuous recording (see the module docstring).

    The trials (``n_trials`` per condition, shuffled) tile the recording; the cue of each trial
    sits ``-epoch[0]`` seconds after the trial start. ``truth`` carries the subject's random
    draws (``depth_scale``, ``erp_scale``, ``latency_shift``), the config, the blink train and
    ``artifact_trials`` (trial index -> ``"burst"``, ``"sway"`` or ``"pop"``).
    """
    cfg = config or SimConfig()
    rng = np.random.default_rng(seed)
    names = tuple(DEFAULT_CHANNELS)
    coords = _coords(names)
    fs = cfg.fs

    depth_cap = 0.95 / cfg.mu_depth if cfg.mu_depth > 0 else 10.0
    depth_scale = float(np.clip(rng.lognormal(0.0, cfg.subject_sd), 0.4, depth_cap))
    erp_scale = float(rng.lognormal(0.0, cfg.subject_sd))
    latency_shift = float(rng.normal(0.0, 0.015))

    n_cond = len(cfg.conditions)
    order = rng.permutation(np.repeat(np.arange(n_cond), cfg.n_trials))
    trial_len = round(cfg.span * fs)
    n = trial_len * len(order)
    cue_offset = round(-cfg.epoch[0] * fs)
    cues = np.arange(len(order)) * trial_len + cue_offset
    tau = cfg.epoch[0] + np.arange(trial_len) / fs  # time relative to the cue inside a trial

    # --- rhythm sources -------------------------------------------------------------
    bg_share = {b: _band_share(BANDS[b], fs) for b in BANDS}
    amp = {b: math.sqrt(cfg.rhythm_snr * cfg.background_uv**2 * bg_share[b]) for b in BANDS}
    centres = {"L": DEFAULT_CHANNELS["C3"], "R": DEFAULT_CHANNELS["C4"]}
    peak = {"mu": cfg.mu_hz, "beta": cfg.beta_hz}
    data = cfg.background_uv * _pink_noise(rng, (len(names), n), fs)
    for rhythm in ("mu", "beta"):
        for hemi, centre in centres.items():
            gain = np.empty(n)
            for k, code in enumerate(order):
                side = _side(cfg.conditions[code])
                g = _gain(cfg, rhythm, hemi, side, tau, depth_scale)
                gain[k * trial_len : (k + 1) * trial_len] = g
            envelope = np.sqrt(np.clip(1.0 + gain, 1e-6, None))
            source = amp[rhythm] * _narrowband(rng, n, fs, peak[rhythm], _NOISE_BANDWIDTH[rhythm])
            weights = _spatial_weights(coords, centre, _SPATIAL_SIGMA)
            data += weights[:, None] * (source * envelope)[None, :]
    occipital = amp["mu"] * _narrowband(rng, n, fs, cfg.mu_hz - 0.5, _NOISE_BANDWIDTH["mu"])
    data += _spatial_weights(coords, DEFAULT_CHANNELS["Oz"], _OCCIPITAL_SIGMA)[:, None] * occipital

    # --- phase-locked ERP -----------------------------------------------------------
    patterns = [  # (centre, sigma) of the topography of N200, P300, N400
        ((0.0, 0.35), 0.55),
        ((0.0, -0.10), 0.70),
        ((0.0, -0.30), 0.70),
    ]
    topo = np.stack([_spatial_weights(coords, c, s) for c, s in patterns])
    for k, code in enumerate(order):
        tags = set(cfg.conditions[code].split("/"))
        trial = np.zeros((len(names), trial_len))
        for c in range(3):
            gain_c = cfg.p300_gain_high if (c == 1 and "high" in tags) else 1.0
            a = cfg.erp_uv[c] * erp_scale * gain_c * (1.0 + 0.25 * rng.standard_normal())
            lat = cfg.erp_latency[c] + latency_shift + 0.015 * rng.standard_normal()
            wave = a * np.exp(-((tau - lat) ** 2) / (2 * cfg.erp_width[c] ** 2))
            trial += topo[c][:, None] * wave[None, :]
        data[:, k * trial_len : (k + 1) * trial_len] += trial

    # --- blinks ---------------------------------------------------------------------
    veog = np.zeros(n)
    n_blinks = int(rng.poisson(cfg.blink_rate * n / fs)) if cfg.blink_rate > 0 else 0
    blink_starts = np.sort(rng.integers(0, max(n - int(0.5 * fs), 1), size=n_blinks))
    for s in blink_starts:
        dur = int(rng.uniform(0.25, 0.40) * fs)
        shape = 0.5 * (1 - np.cos(2 * np.pi * np.arange(dur) / dur))
        veog[s : s + dur] += cfg.blink_uv * rng.lognormal(0.0, 0.2) * shape
    blink_topo = _spatial_weights(coords, (0.0, 1.15), 0.45)
    data += blink_topo[:, None] * veog[None, :]

    # --- artefacts in a fraction of the trials ---------------------------------------
    artifact_trials: dict[int, str] = {}
    for k in range(len(order)):
        if rng.random() >= cfg.artifact_rate:
            continue
        kind = ("burst", "sway", "pop")[int(rng.integers(3))]
        artifact_trials[k] = kind
        base = k * trial_len
        if kind == "burst":  # broadband muscle activity on 1-3 channels, about 1 s
            chans = rng.choice(len(names), size=int(rng.integers(1, 4)), replace=False)
            dur = int(rng.uniform(0.8, 1.2) * fs)
            start = base + int(rng.uniform(0.3, cfg.span - 1.6) * fs)
            data[chans, start : start + dur] += cfg.artifact_uv * rng.standard_normal(
                (len(chans), dur)
            )
        elif kind == "sway":  # 1.5 Hz wave of 100-200 uV on one channel for 2.5 s
            chan = int(rng.integers(len(names)))
            dur = int(2.5 * fs)
            start = base + int(rng.uniform(0.3, cfg.span - 3.0) * fs)
            wave = np.sin(2 * np.pi * 1.5 * np.arange(dur) / fs + rng.uniform(0, 2 * np.pi))
            data[chan, start : start + dur] += rng.uniform(120, 200) * wave
        else:  # electrode pop: a sharp 200-300 uV deflection that decays in ~0.3 s
            chan = int(rng.integers(len(names)))
            start = base + int(rng.uniform(0.3, cfg.span - 1.0) * fs)
            decay = int(0.3 * fs)
            sign = 1.0 if rng.random() < 0.5 else -1.0
            data[chan, start : start + decay] += (
                sign * rng.uniform(200, 300) * np.exp(-np.arange(decay) / (0.05 * fs))
            )

    # --- line noise -----------------------------------------------------------------
    t_all = np.arange(n) / fs
    phases = rng.uniform(0, 2 * np.pi, size=len(names))
    data += cfg.line_uv * np.sin(2 * np.pi * cfg.line_hz * t_all[None, :] + phases[:, None])

    eog = veog + 2.0 * bandpass(rng.standard_normal(n), fs, 0.5, 10.0, order=2)
    events = np.column_stack([cues, order + 1])
    labels = {i + 1: c for i, c in enumerate(cfg.conditions)}
    truth = {
        "config": cfg,
        "depth_scale": depth_scale,
        "erp_scale": erp_scale,
        "latency_shift": latency_shift,
        "blink_starts": blink_starts,
        "blink_signal": veog * 1e-6,
        "artifact_trials": artifact_trials,
    }
    return Recording(
        data=data * 1e-6,
        fs=fs,
        ch_names=names,
        events=events,
        event_labels=labels,
        eog=eog * 1e-6,
        subject=subject,
        truth=truth,
    )


def simulate_group(
    n_subjects: int,
    config: SimConfig | None = None,
    seed: int = 0,
) -> list[Recording]:
    """``n_subjects`` independent recordings named ``S01``, ``S02``, ... (reproducible)."""
    if n_subjects < 1:
        raise ValueError("n_subjects must be at least 1")
    children = np.random.SeedSequence(seed).spawn(n_subjects)
    width = max(2, len(str(n_subjects)))
    return [
        simulate_recording(config, child, subject=f"S{i + 1:0{width}d}")
        for i, child in enumerate(children)
    ]
