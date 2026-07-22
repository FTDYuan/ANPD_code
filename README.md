# ANPD: Addiction Neural Pattern Disentanglement Network

Official PyTorch implementation of **ANPD**, a neural representation framework for characterizing shared, type-specific, and individual-level patterns in addiction-related resting-state EEG.

> **Manuscript:** *An Addiction Neural Pattern Disentanglement Network Reframes Addiction Heterogeneity*

## Overview

Addiction-related conditions share neural characteristics across substances and behaviors while also showing marked variation across addiction types and individuals. ANPD models these complementary sources of variation within a unified representation space.

The framework combines source-level EEG activity and inter-regional functional coupling to learn:

- a **shared representation** capturing neural organization conserved across addiction types;
- a **type-specific representation** capturing addiction-type-dependent neural configurations;
- an **individual neural state** describing the relative expression of shared and type-specific patterns.

The resulting **Addiction Neural Contribution Index (ANCI)** places each participant along a shared-to-type-specific neural axis and supports analyses of individualized neural states, symptom severity, and type-dependent network expression.

## Framework

The main pipeline consists of:

1. **EEG preprocessing and source representation**  
   Resting-state EEG is filtered, cleaned, segmented, and projected to 116 regions of the Automated Anatomical Labeling atlas.

2. **Temporal and network feature construction**  
   Band-specific regional signals preserve local temporal information, while weighted phase-lag index connectivity characterizes inter-regional functional coupling.

3. **Shared and type-specific graph encoders**  
   Parallel graph-attention branches learn cross-addiction and addiction-type-dependent representations.

4. **Type-guided modulation**  
   A learned type embedding modulates the specific branch while leaving the shared branch unaffected.

5. **Addiction neural pattern gating**  
   A node-wise gate interpolates between shared and type-specific representations and produces the fused representation used for prediction.

6. **Individualized neural-state analysis**  
   ANCI quantifies the relative shift from shared toward type-specific neural expression for each participant.

## Main Outputs

ANPD supports:

- multi-class classification across addiction-related phenotypes and healthy controls;
- identification of a cross-addiction neural core;
- characterization of type-specific network configurations;
- estimation of individualized neural states using ANCI;
- association analyses between neural representations and symptom severity;
- validation in independent internet-, food-, and gaming-related cohorts.

## Repository Structure

The repository is expected to follow the structure below. Update the file names in this section if your released code uses a different layout.

```text
ANPD/
├── configs/                 # Training, preprocessing, and evaluation settings
├── data/                    # Dataset metadata and example input structure
├── models/                  # ANPD model components
│   ├── temporal_encoder.py
│   ├── graph_encoder.py
│   ├── type_embedding.py
│   ├── gating.py
│   └── anpd.py
├── preprocessing/           # EEG preprocessing and network construction
├── analysis/                # ANCI, network attribution, and visualization
├── scripts/                 # Training and evaluation entry points
├── checkpoints/             # Trained model weights
├── requirements.txt
└── README.md
```

## Environment

A CUDA-enabled GPU is recommended. The experiments reported in the manuscript were conducted using PyTorch on an NVIDIA A6000 GPU with 48 GB memory.

```bash
conda create -n anpd python=3.10
conda activate anpd
pip install -r requirements.txt
```

The principal dependencies include:

- Python
- PyTorch
- NumPy
- SciPy
- scikit-learn
- pandas
- MNE-Python
- NetworkX
- matplotlib

Exact package versions should be recorded in `requirements.txt` or `environment.yml`.

## Data Preparation

### Expected processing pipeline

The analyses in the manuscript use the following general preprocessing workflow:

```text
Raw resting-state EEG
    ↓
0.1–45 Hz filtering
    ↓
Resampling to 250 Hz
    ↓
Re-referencing and artifact removal
    ↓
10-s segmentation
    ↓
Source projection to AAL-116 regions
    ↓
Band-specific regional signals
    ↓
wPLI functional-connectivity matrices
```

To prevent information leakage, all segments from the same participant must remain in the same training, validation, or test split.

### Input organization

The exact input schema depends on the released preprocessing scripts. At minimum, each sample should include:

```text
subject identifier
addiction-type label
regional EEG representation
functional-connectivity matrix
clinical or symptom score, when available
dataset or cohort identifier
```

Do not upload restricted participant-level EEG or identifiable clinical data to this repository.

## Data Sources

The repository does not redistribute the original EEG datasets. Users must obtain each dataset from its original source and comply with the corresponding data-use terms.

| Cohort | Source | Access |
|---|---|---|
| Locally acquired DUD and control data | Guangzhou Cengcun Drug Rehabilitation Center and Guangzhou Women’s Compulsory Isolation Drug Rehabilitation Center | Restricted; access may be requested from the corresponding authors, subject to ethics and institutional approval |
| IA01 and AA01 | MPI Leipzig Mind-Brain-Body Dataset (MPI-LEMON) | https://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html |
| IA02 | Qi et al., 2022 | Available from the corresponding author of the original study upon reasonable request: https://doi.org/10.3390/ijerph19052686 |
| IA03 and FA01 | Healthy Brain Network | https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/ |
| GA01 | MOBA EEG dataset, OpenNeuro accession `ds005520` | https://openneuro.org/datasets/ds005520 |

For the Healthy Brain Network, EEG data and limited demographic information are openly accessible, whereas complete phenotypic data require completion of the relevant Data Usage Agreement.

## Training

Replace the placeholders below with the actual script and configuration names used in the released repository.

```bash
python scripts/train.py \
    --config configs/anpd.yaml \
    --data_dir /path/to/processed_data \
    --output_dir outputs/anpd
```

Recommended evaluation should use participant-level cross-validation rather than segment-level random splitting.

Key training settings, including the random seed, learning rate, batch size, number of folds, frequency band, graph construction parameters, and loss weights, should be stored in the configuration file.

## Evaluation

```bash
python scripts/evaluate.py \
    --config configs/anpd.yaml \
    --checkpoint outputs/anpd/best_model.pt \
    --data_dir /path/to/processed_data
```

Recommended metrics include:

- accuracy;
- balanced accuracy;
- macro F1 score;
- class-wise sensitivity and specificity;
- receiver operating characteristic area under the curve;
- confidence intervals or variation across folds and random seeds.

## Representation and ANCI Analysis

```bash
python analysis/analyze_representations.py \
    --checkpoint outputs/anpd/best_model.pt \
    --data_dir /path/to/processed_data \
    --output_dir outputs/representation_analysis
```

The analysis should export, where applicable:

```text
shared embeddings
type-specific embeddings
node-wise gating values
subject-level ANCI values
network attribution scores
clinical association statistics
```

Higher ANCI values indicate a stronger shift toward type-specific neural expression, whereas lower values indicate greater dominance of the shared representation.

## External Validation

External validation should freeze the pretrained representation model unless a different experimental setting is explicitly specified. Only the designated prediction head or downstream classifier should be trained on the external cohort.

```bash
python scripts/external_validation.py \
    --config configs/external_validation.yaml \
    --checkpoint outputs/anpd/best_model.pt \
    --data_dir /path/to/external_dataset \
    --cohort IA03
```

Supported external analyses may include the IA03, FA01, and GA01 cohorts.

## Reproducibility

For reproducible experiments:

- use participant-level data splitting;
- fix and report all random seeds;
- retain the exact preprocessing parameters;
- record software and CUDA versions;
- save fold assignments;
- report results across folds or repeated runs;
- avoid selecting preprocessing or model settings using the external test cohorts.

## Pretrained Models

Pretrained checkpoints will be provided in the `checkpoints/` directory or through the repository release page.

```text
checkpoints/
├── anpd_full.pt
├── shared_encoder.pt
└── [additional checkpoints]
```

Update this section with the final checkpoint names and download links before public release.

## Citation

The manuscript is currently under review. Please use the following temporary citation and replace it with the final journal citation after publication:

```bibtex
@article{yuan2026anpd,
  title   = {An Addiction Neural Pattern Disentanglement Network Reframes Addiction Heterogeneity},
  author  = {Yuan, Haozhang and Chen, Bianna and Bao, Yanping and Guo, Jifeng and Shen, Shu and Chen, C. L. Philip and Zhang, Tong},
  journal = {Manuscript under review},
  year    = {2026}
}
```

## Ethics and Responsible Use

This repository is intended for research use. ANPD outputs should not be interpreted as standalone clinical diagnoses or treatment recommendations. Any use of participant-level EEG or clinical data must follow the original informed-consent scope, ethics approval, institutional requirements, and dataset-specific licenses.

## License

Add the selected open-source license before release, for example:

```text
MIT License
```

The license for this code does not override the licenses or data-use restrictions of the original datasets.

## Contact

For questions about the code or data-access procedures, please contact the corresponding authors:

- **C. L. Philip Chen:** Philip.Chen@ieee.org
- **Tong Zhang:** tony@scut.edu.cn
