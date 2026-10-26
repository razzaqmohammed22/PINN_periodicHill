# Physics-Informed Neural Networks for Periodic Hill Flow

## Project Description

This project explores **Physics-Informed Neural Networks (PINNs)** for modelling fluid flow over periodic hills. The neural network incorporates the Navier–Stokes equations into its training process to learn flow-field variables while respecting governing physical principles.

The model uses reference CFD data for training and validation, enabling comparison between PINN predictions and numerical simulation results.

## Project Structure

```text
PINN_periodicHill/
├── data/
│   ├── komegasst_PHLL_case_1p0_Cx.npy
│   ├── komegasst_PHLL_case_1p0_Cy.npy
│   ├── komegasst_PHLL_case_1p0_Ux.npy
│   ├── komegasst_PHLL_case_1p0_Uy.npy
│   ├── komegasst_PHLL_case_1p0_p.npy
│   └── komegasst_PHLL_case_1p0_gradU.npy
├── npy-matlab/
│   ├── constructNPYheader.m
│   ├── datToNPY.m
│   ├── readNPY.m
│   ├── readNPYheader.m
│   └── writeNPY.m
├── Periodic_Hill_PINN.m
├── README.md
└── LICENSE
```

### Directory and File Descriptions

- **`data/`** — Contains NumPy (`.npy`) files with periodic hill flow coordinates, velocity components, pressure, and velocity-gradient data.
- **`npy-matlab/`** — MATLAB utilities for reading and writing NumPy files.
- **`Periodic_Hill_PINN.m`** — Main MATLAB script for training and evaluating the PINN model.
- **`README.md`** — Project documentation and setup instructions.
- **`LICENSE`** — Project licensing information.

## Dependencies and Requirements

Ensure that the following software and toolboxes are installed:

- MATLAB R2022a or later
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox
- `npy-matlab` library (included in this repository)

## Installation and Setup

### Step 1: Clone the Repository

Open a terminal or Git Bash and navigate to your preferred working directory.

For example, on Windows:

```bash
cd "C:\Users\Razzaq\Documents"
```

Clone the repository:

```bash
git clone https://github.com/razzaqmohammed22/PINN_periodicHill.git
```

Navigate to the project directory:

```bash
cd PINN_periodicHill
```

### Step 2: Open the Project in MATLAB

1. Launch MATLAB.
2. Navigate to the cloned `PINN_periodicHill` directory using the **Current Folder** panel.
3. Verify that `Periodic_Hill_PINN.m` and the `data/` directory are present.

Alternatively, open `Periodic_Hill_PINN.m` directly from the project folder.

### Step 3: Run the Main Script

Run the following command in the MATLAB Command Window:

```matlab
Periodic_Hill_PINN
```

Ensure that the current working directory is the project root before executing the script.

## Expected Outputs

After running the main script, the program is expected to report:

- Training and validation losses across epochs
- PINN training time
- Comparison table for 10 validation samples
- Mean Absolute Error (MAE) for the full dataset
- MAE for the validation dataset
- Root Mean Squared Error (RMSE) for the full dataset
- Analysis summary
- Three figures and an additional plot

*Actual outputs depend on the dataset, model configuration, and successful execution of the script.*

## Troubleshooting

### File Not Found Error

If MATLAB reports that a file cannot be found:

1. Check that the current working directory is the project root.
2. Confirm that all required `.npy` files are present in the `data/` directory.
3. Verify that the script uses the correct relative paths.
4. Ensure that the `npy-matlab/` directory is available on the MATLAB path.

### Missing Toolbox

If MATLAB reports an unavailable function, verify that the required toolbox is installed and licensed.

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
