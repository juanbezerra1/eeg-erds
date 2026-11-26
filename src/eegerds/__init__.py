"""eeg-erds: ERD/ERS, ERP peaks, SOBI blink removal and epoch rejection for epoched EEG."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .artifacts import RejectionConfig, RejectionReport, reject_epochs
from .data import Epochs, Recording, epoch
from .erds import (
    ErdsConfig,
    ErdsCurve,
    GroupErds,
    erds_curve,
    erds_group,
    sample_power,
)
from .erp import (
    DEFAULT_COMPONENTS,
    GroupErp,
    Peak,
    PeakSpec,
    find_group_peaks,
    find_peaks_ordered,
    group_erp,
    measure_amplitudes,
)
from .features import features_table, trial_features
from .filters import bandpass, bandpass_recording, lowpass, notch
from .ica import EogRemoval, SobiResult, remove_eog_components, sobi
from .io import from_mne, load_epochs, save_epochs, to_mne
from .pipeline import PipelineConfig, SubjectResult, process_group, process_recording
from .stats import (
    compare_amplitudes,
    compare_channels,
    compare_conditions,
    compare_series,
    fdr_bh,
    paired_wilcoxon,
    signflip_cluster,
    signflip_maxt,
)
from .synth import SimConfig, expected_erds, simulate_group, simulate_recording

try:
    __version__ = version("eeg-erds")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0+local"

__all__ = [
    "DEFAULT_COMPONENTS",
    "Epochs",
    "EogRemoval",
    "ErdsConfig",
    "ErdsCurve",
    "GroupErds",
    "GroupErp",
    "Peak",
    "PeakSpec",
    "PipelineConfig",
    "Recording",
    "RejectionConfig",
    "RejectionReport",
    "SimConfig",
    "SobiResult",
    "SubjectResult",
    "__version__",
    "bandpass",
    "bandpass_recording",
    "compare_amplitudes",
    "compare_channels",
    "compare_conditions",
    "compare_series",
    "epoch",
    "erds_curve",
    "erds_group",
    "expected_erds",
    "features_table",
    "fdr_bh",
    "find_group_peaks",
    "find_peaks_ordered",
    "from_mne",
    "group_erp",
    "load_epochs",
    "lowpass",
    "measure_amplitudes",
    "notch",
    "paired_wilcoxon",
    "process_group",
    "process_recording",
    "reject_epochs",
    "remove_eog_components",
    "sample_power",
    "save_epochs",
    "signflip_cluster",
    "signflip_maxt",
    "simulate_group",
    "simulate_recording",
    "sobi",
    "to_mne",
    "trial_features",
]
