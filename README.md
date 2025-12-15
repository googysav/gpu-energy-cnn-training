# Analysis of Cumulative GPU Energy Consumption Across CNN Architecture Models
This repository contains the full experimental code and visualisation pipeline accompanying the report:

“Analysis of Cumulative GPU Energy Consumption Across CNN Architecture Models” 

The project investigates how CNN feature channel width and numerical precision (FP32 vs mixed precision) affect training performance and estimated GPU energy consumption.  

# Project Overview:
Modern CNN training prioritises accuracy, often overlooking computational energy cost. This project explores how architectural scale and precision choices influence energy efficiency during training.

The key notions addressed include:
-
- How increased CNN channel width/depth affects accuracy, loss convergence, and energy use
- The benefits and detriments of mixed-precision on lightweight CNN architectures
- The optomisation of energy-accuracy metrics by altering width and precision modes
- 
Experiments are conducted on a lightweight CNN trained on CIFAR-10 using PyTorch, with energy estimated from training time under a fixed GPU power envelope (NVIDIA T4, 70W).

# Summary of Findings:
Key conclusions from the experiments include:
-
- Increasing CNN width consistently improves accuracy and loss convergence.
- Energy consumption increases proportionally with training time.
- Mixed precision training reduces energy consumption only for the largest model width.
- For smaller CNNs, mixed precision provides little to no benefit and can slightly increase energy usage.
- Energy efficiency gains from mixed precision are strongly dependent on model scale and hardware utilisation.
- Model optimisation should consider the its context, as optimisation parameters may scale.

These findings are discussed in detail in the accompanying report.

# Repository Structure
```text
├── src/
│   ├── data.py            # CIFAR-10 loading and preprocessing
│   ├── model.py           # CNN architecture definition
│   ├── training_eval.py   # Training and validation logic
│   ├── experiment.py      # Experiment orchestration and metric tracking
│   └── metrics.py         # Dataclass for storing results
│
├── scripts/
│   ├── main.py            # Entry point for running experiments
│   └── plots.py           # Result visualisation
│
├── results/               # Generated JSON experiment outputs
├── figures/               # Generated plots
└── README.md
```

# Reproducibility
Full Experiment (GPU)
-
The full experiment was run on Google Colab with GPU acceleration due to computational cost and time restraints.
- GPU: NVIDIA T4 (70W)
- Epochs: 30 (hard coded)
- Dataset CIFAR-10
- Precision modes: FP32, AMP
- Widths: 1, 2, 3 (scalars)

This configuration reproduces the results reported in the paper.

Local Excecution (CPU)
-
The code is fully functional on CPU for verification and development purposes.
These are intended as smoke tests, and do not reproduce GPU energy values due to speed and hardware differences.

# Notes on Energy Estimation
- Energy is calculated as:
```text
Energy (kWh) = GPU Power (W) × Time (hours) / 1000
```
- GPU power output is assumed constant at 70W (NVIDIA T4 Specifications)
- This enables comparative analysis but supresses any results surrounding transient power variation
Limitations and improvements are discussed in the report.

# Report
The full report discussing theory, methodology, results, discussion, and final conclusion is available here:
[Analysis of Cumulative GPU Energy Consumption Across CNN Architecture Models (PDF)] 


