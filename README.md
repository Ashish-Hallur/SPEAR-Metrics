# SPEAR-Metrics

SPEAR-Metrics contains the extraction and analysis code used for large-scale speech-native evaluation baselines over the Seamless Interaction corpus. It focuses on conversational signals that text-only evaluation misses: pitch behavior, speaking rhythm, pausing, and lexical properties of ASR-aligned transcripts.

The accompanying manuscript is included as `InterSpeech_Submission.pdf` and `final.tex`:

> Distributional Baselines for Conversational Prosody and Rhythm  
> Ashish G. Hallur, Thomas Thebaud, Venkatesh Ravichandran, Georgi Tinchev, and Laureano Moro-Velazquez

The paper analyzes 4,000+ hours of dyadic English conversation and derives reference operating regimes for prosodic and temporal behavior. This repository also includes lexical extraction and exploratory lexical notebooks that go beyond the main paper narrative.

## What This Repository Does

- Extracts robust fundamental-frequency, or F0, summaries from waveform files.
- Extracts temporal rhythm features from word-level timestamps and VAD segments.
- Extracts lexical density, diversity, distributional, and discourse features from ASR transcript JSON files.
- Provides notebooks for diagnostics, demographic and interaction-state analyses, and publication figure generation.
- Documents empirical operating regimes that can be reused as reference distributions for speech-to-speech system evaluation.

## Repository Layout

```text
.
|-- README.md
|-- LICENSE
|-- InterSpeech_Submission.pdf
|-- final.tex
|-- all_wavs.txt
|-- master_figures_V5.ipynb
|-- Prosodic/
|   |-- run_f0_extraction.py
|   |-- f0_extraction.slurm
|   `-- Analyses/
|       |-- 01_methodology_and_diagnostics.ipynb
|       |-- 02_demographics_age_gender.ipynb
|       |-- 03_social_context_and_relationships.ipynb
|       `-- 04_interactional_dynamics_and_personality.ipynb
|-- Temporal/
|   |-- run_temporal_extraction.py
|   |-- temporal_extraction.slurm
|   `-- 07_temporal_analysis.ipynb
`-- Lexical/
    |-- run_lexical_extraction.py
    |-- lexical_extraction.slurm
    |-- all_jsons.txt
    `-- Analyses/
        |-- 05_lexical_analysis.ipynb
        `-- 06_lexical_demographic_analysis.ipynb
```

## Data Assumptions

This repository does not ship the Seamless Interaction audio, transcript JSON, VAD metadata, Vox-Profile outputs, or generated master CSV files. The extraction scripts were written for the original lab filesystem layout and contain hard-coded paths such as:

- `/export/fs06/corpora8/seamless_interaction/datasets/...`
- `/home/ahallur1/spear/Seamless_Experiments/...`
- `/home/ahallur1/spear/Vox_Profile/vox-profile-release/...`

To run the pipeline elsewhere, either mirror that directory structure or edit the path constants at the top of the scripts and notebooks.

Expected inputs:

- WAV files listed in `all_wavs.txt`.
- Seamless Interaction JSON transcript files listed in `Lexical/all_jsons.txt`.
- JSON transcripts with `metadata:transcript` word timings and `metadata:vad` VAD segments.
- Optional relationship metadata from `relationships.csv`.
- Optional Vox-Profile file-level annotations for predicted age, sex/gender label, arousal, valence, dominance, and personality-related analyses.

## Environment

The code is Python-based and was run in a conda environment on a SLURM cluster. Main packages used across scripts and notebooks include:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `tqdm`
- `parselmouth` / Praat
- `spacy`
- `jupyter`

Install the spaCy model before lexical extraction:

```bash
python -m spacy download en_core_web_sm
```

## Pipeline Overview

The extraction pipeline is sharded. Each extractor takes a shard index and a total number of shards, then writes one CSV per shard. The supplied SLURM files use 431 array jobs.

### 1. Prosodic Extraction

Script: `Prosodic/run_f0_extraction.py`  
SLURM wrapper: `Prosodic/f0_extraction.slurm`

This script reads WAV paths, extracts pitch tracks with Praat's autocorrelation F0 estimator through `parselmouth`, and writes robust F0 summaries.

Main settings:

- F0 floor: `75 Hz`
- F0 ceiling: `500 Hz`
- Minimum voiced ratio flag: `0.05`
- Robust trims: `10-90%` and `25-75%`

Important output columns:

- Identifiers: `wav_path`, `orig_id`, `vendor_id`, `session_id`, `subset`, `split`
- Relationship metadata: `relationship`, `relationship_detail`
- Durations: `total_duration_s`, `voiced_duration_s`, `voiced_ratio`, `n_voiced_frames`
- Raw F0: `f0_mean_raw`, `f0_median_raw`, `f0_std_raw`, `f0_min_raw`, `f0_max_raw`, `f0_range_raw`
- Robust F0: `f0_p10`, `f0_p90`, `f0_range_p10_p90`, `f0_mean_p10_p90`, `f0_std_p10_p90`, plus corresponding `25-75%` fields
- Status: `OK`, `NO_VOICED_FRAMES`, `LOW_VOICED_RATIO`, or `ERROR: ...`

Run locally:

```bash
python Prosodic/run_f0_extraction.py 0 431
```

Run on SLURM:

```bash
sbatch Prosodic/f0_extraction.slurm
```

### 2. Temporal Extraction

Script: `Temporal/run_temporal_extraction.py`  
SLURM wrapper: `Temporal/temporal_extraction.slurm`

This script computes rhythm metrics from ASR word timings and VAD segments. It uses VAD to define speech-activity stretches, merges nearby VAD segments, keeps only stable continuous stretches, and computes rates and pause statistics on the retained material.

Main settings:

- Pause threshold: `0.2 s`
- VAD merge gap: `1.0 s`
- Minimum retained speech stretch duration: `12.1 s`

Important output columns:

- `orig_id`, `wav_path`
- `total_duration_s`
- `speech_active_time_s`
- `pause_count`
- `pause_total_duration_s`
- `pause_mean_duration_s`
- `pause_ratio`
- `speech_rate_wps`, `speech_rate_wpm`
- `articulation_rate_wps`, `articulation_rate_wpm`
- Status values such as `OK`, `TOO_FEW_WORDS`, `NO_VAD`, `NO_VALID_STRETCH`, `INSUFFICIENT_TIMED_WORDS_IN_STRETCHES`, or `ERROR: ...`

Metric definitions:

- `speech_rate_wpm = 60 * words / total_retained_stretch_time`
- `articulation_rate_wpm = 60 * words / speech_active_time`
- `speech_active_time = total_retained_stretch_time - pause_time`
- `pause_ratio = pause_time / total_retained_stretch_time`

Run locally:

```bash
python Temporal/run_temporal_extraction.py --shard_idx 0 --num_shards 431
```

Optional flags:

```bash
python Temporal/run_temporal_extraction.py \
  --all_jsons_txt /path/to/all_jsons.txt \
  --out_dir /path/to/temporal/shard_csvs \
  --shard_idx 0 \
  --num_shards 431 \
  --pause_threshold_s 0.2 \
  --merge_gap_threshold_s 1.0 \
  --min_stretch_duration_s 12.1
```

Run on SLURM:

```bash
sbatch Temporal/temporal_extraction.slurm
```

### 3. Lexical Extraction

Script: `Lexical/run_lexical_extraction.py`  
SLURM wrapper: `Lexical/lexical_extraction.slurm`

This script reads transcript JSON files and computes lexical features from normalized ASR words. It uses spaCy POS tags for content/function word counts.

Main settings:

- MATTR small window: `50`
- MATTR large window: `500`
- MTLD threshold: `0.72`
- Minimum words for lexical diversity: `50`
- Low ASR confidence threshold: `0.7`

Important output columns:

- Identifiers: `orig_id`, `wav_path`
- Status: `lexical_status`, `status_reason`
- Size and confidence: `total_words`, `unique_words`, `mean_asr_confidence`, `low_conf_flag`
- Lexical density: `content_word_count`, `function_word_count`, `lexical_density`
- Diversity: `ttr`, `mattr_small`, `mattr_large`, `mattr_ratio`, `mtld`
- Distribution shape: `hapax_ratio`, `lexical_entropy`
- Discourse: `backchannel_ratio`, `discourse_marker_ratio`

Run locally:

```bash
python Lexical/run_lexical_extraction.py 0 431
```

Run on SLURM:

```bash
sbatch Lexical/lexical_extraction.slurm
```

## Combining Shards

The repository does not include a dedicated shard-combining script. In the original workflow, per-shard CSVs were concatenated into master files consumed by the notebooks, including:

- `/home/ahallur1/spear/Seamless_Experiments/F0/seamless_f0_features.csv`
- `/home/ahallur1/spear/Seamless_Experiments/Temporal/temporal_master.csv`
- `/home/ahallur1/spear/Seamless_Experiments/Lexical/seamless_lexical_features.csv`
- `/home/ahallur1/spear/Seamless_Experiments/Master/seamless_features.csv`

Example concatenation:

```python
from pathlib import Path
import pandas as pd

src = Path('/path/to/shard_csvs')
out = Path('/path/to/master.csv')
dfs = [pd.read_csv(p) for p in sorted(src.glob('*.csv'))]
pd.concat(dfs, ignore_index=True).to_csv(out, index=False)
```

## Analysis Notebooks

The notebooks assume the generated master CSVs already exist at the original analysis paths. Update the `Path(...)` cells if you are using a different layout.

### Master Figures

`master_figures_V5.ipynb` reproduces the final prosodic and temporal figures used in the paper. It loads `Seamless_Experiments/Master/seamless_features.csv`, applies consistent filtering, and saves high-resolution PNG and PDF versions of:

- Fig. 1: Prosody by gender
- Fig. 2: Prosody by arousal sextile
- Fig. 3: Prosody by dominance sextile
- Fig. 4: Temporal trends by arousal and dominance
- Fig. 5: Age effects on prosody and rhythm

### Prosodic Analyses

- `Prosodic/Analyses/01_methodology_and_diagnostics.ipynb`: validates robust F0 estimation, voiced-ratio filtering, and raw vs trimmed F0 behavior.
- `Prosodic/Analyses/02_demographics_age_gender.ipynb`: analyzes predicted age and gender effects on pitch metrics.
- `Prosodic/Analyses/03_social_context_and_relationships.ipynb`: explores familiar vs stranger interactions, relationship details, and conversation-level pitch stability.
- `Prosodic/Analyses/04_interactional_dynamics_and_personality.ipynb`: relates F0 metrics to Vox-Profile emotional stance, dominance, valence, and personality-related variables.

### Temporal Analysis

`Temporal/07_temporal_analysis.ipynb` loads temporal metrics, merges them with the master feature table, and produces paper-style temporal plots for speech rate and pause ratio.

### Lexical Analyses

- `Lexical/Analyses/05_lexical_analysis.ipynb`: explores lexical feature distributions.
- `Lexical/Analyses/06_lexical_demographic_analysis.ipynb`: merges lexical metrics with Vox-Profile annotations for demographic analysis.

## Paper Summary

The paper studies conversational operating regimes for speech-to-speech evaluation. Its main argument is that naturalness cannot be judged only from text, task success, or global subjective scores. Conversational speech systems should also be evaluated against speech-native reference distributions for prosody and rhythm.

Dataset:

- Seamless Interaction dyadic English conversations.
- 4,065.04 hours of interaction time.
- 64,739 interaction segments from 5,098 one-hour recording sessions.
- 4,284 participants.
- Metrics are computed at the speaker-channel level.

Prosodic metrics:

- F0 mean, standard deviation, and range.
- Praat autocorrelation F0 extraction through parselmouth.
- Conservative 75-500 Hz bounds.
- Speaker channels with voiced ratio below 0.05 are excluded from paper regimes.
- Reported F0 statistics use 10-90% within-channel trimming to reduce tracking artifacts and outliers.

Temporal metrics:

- Speech rate, articulation rate, pause ratio, and mean pause duration.
- Word timings come from ASR-aligned transcripts.
- Speech-activity stretches come from VAD.
- Adjacent VAD segments are merged if separated by at most 1.0 s.
- Retained stretches must be at least 12.1 s.
- Pauses are inter-word gaps of at least 0.2 s.

Vox-Profile annotations:

- Predicted age and binary sex/gender label.
- Continuous arousal, valence, and dominance scores in `[0, 1]`.
- These are model-derived annotations, not ground-truth self-reports.

## Reported Operating Regimes

Pooled speaker-channel regimes reported in the paper:

| Track | Metric | Median | IQR | Mean |
| --- | --- | ---: | --- | ---: |
| Prosody, N=121,813, 3,863 h | F0 mean, 10-90% trimmed, Hz | 157.4 | 120.1-198.6 | 161.5 |
| Prosody, N=121,813, 3,863 h | F0 SD, 10-90% trimmed, Hz | 20.84 | 13.79-30.07 | 23.22 |
| Prosody, N=121,813, 3,863 h | F0 range, 10-90% trimmed, Hz | 87.11 | 57.11-125.8 | 95.82 |
| Temporal, N=91,471, 3,045 h | Speech rate, wpm | 175.9 | 156.0-195.9 | 175.8 |
| Temporal, N=91,471, 3,045 h | Articulation rate, wpm | 237.8 | 216.1-259.5 | 237.2 |
| Temporal, N=91,471, 3,045 h | Pause ratio | 0.2575 | 0.2166-0.2996 | 0.2595 |
| Temporal, N=91,471, 3,045 h | Mean pause duration, s | 0.5845 | 0.5225-0.6559 | 0.6058 |

Selected stratified findings:

- Mean F0 is strongly conditioned by predicted gender, so pooled absolute-F0 targets are not appropriate for evaluation.
- F0 SD and F0 range increase monotonically with predicted arousal and dominance.
- Speech rate increases with arousal and dominance.
- Pause ratio decreases with arousal and dominance.
- Age bins are associated with smaller but measurable shifts in F0 mean, speech rate, and pause ratio.

Selected effect sizes reported in the paper:

| Factor | Metric | Effect |
| --- | --- | ---: |
| Gender | Mean F0 | Cliff's delta = -0.957 |
| Gender | F0 SD | Cliff's delta = -0.635 |
| Gender | F0 range | Cliff's delta = -0.644 |
| Arousal | F0 SD | Spearman rho = 0.544 |
| Arousal | F0 range | Spearman rho = 0.516 |
| Arousal | Speech rate | Spearman rho = 0.187 |
| Arousal | Pause ratio | Spearman rho = -0.170 |
| Dominance | F0 SD | Spearman rho = 0.463 |
| Dominance | F0 range | Spearman rho = 0.430 |
| Dominance | Speech rate | Spearman rho = 0.200 |
| Dominance | Pause ratio | Spearman rho = -0.194 |
| Age bin | Mean F0 | Kruskal-Wallis epsilon squared = 0.00643 |
| Age bin | Speech rate | Kruskal-Wallis epsilon squared = 0.01623 |
| Age bin | Pause ratio | Kruskal-Wallis epsilon squared = 0.00511 |

## Using These Metrics for S2S Evaluation

For a generated speech response or dialogue segment:

1. Compute the same F0, temporal, and optionally lexical metrics.
2. Apply the same quality filters used in this repository.
3. Select a matched reference stratum when metadata or model-derived labels are available, for example predicted gender, age bin, arousal, or dominance.
4. Compare the generated output to the corresponding distribution using percentiles, deviations from the median, or out-of-range flags.
5. Inspect individual cues before collapsing to one score: unnaturalness may come from compressed pitch expressivity, mismatched pace, atypical pausing, or a combination of cues.

## Limitations

- The paper analysis is descriptive and does not yet map metric deviations to perceptual thresholds or human naturalness ratings.
- The operating regimes are based on one English dyadic corpus and may shift by language, domain, recording setup, and interaction setting.
- Age, arousal, dominance, and sex/gender labels are derived from Vox-Profile models. They should be interpreted as model-conditioned annotations unless validated against ground-truth metadata.
- The paper uses a binary sex/gender label predicted from voice and does not model non-binary gender identities.
- The scripts contain hard-coded lab paths and may need light path refactoring before reuse on a new machine.
- The notebooks may contain executed outputs from the original environment; for clean reproduction, restart kernels and rerun after updating paths.

## License

This project is released under the MIT License. See `LICENSE`.
