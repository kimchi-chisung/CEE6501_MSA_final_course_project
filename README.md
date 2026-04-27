# CEE6501 Final Course Project  
## Mixed 3D Truss–Frame–Cable Structural Analysis Solver

This repository contains a Python-based direct stiffness method (DSM) solver for mixed structural systems consisting of 3D truss, 3D frame, and cable elements. The main workflow is centered on `main.ipynb`, where users can load different input models, run the analysis, and review the numerical and graphical results.

## Repository structure

```text
helpers/
inputs/
outputs/
presentation/
report/
validation_ref/
input_generation.ipynb
main.ipynb
requirements.txt
```

### Folder roles

- `helpers/`  
  Contains the main Python modules used by the solver, including preprocessing, element formulation, assembly, solver logic, release handling, fixed-end-force calculation, and postprocessing.

- `inputs/`  
  Stores JSON input files for validation problems and final bridge models. Users can switch between these files in `main.ipynb` to test different cases.

- `outputs/`  
  Stores generated analysis results such as Excel tables, saved plots, and other exported output files.

- `presentation/`  
  Contains presentation materials for the final project.

- `report/`  
  Contains the written report files and related materials.

- `validation_ref/`  
  Contains handwritten or reference solutions used to validate the code output.

- `input_generation.ipynb`  
  Used to generate or edit larger structural input files, especially for the final bridge case study.

- `main.ipynb`  
  Main notebook for running the structural analysis, exporting tables, and generating plots.

- `requirements.txt`  
  Lists the Python packages required to run the project.

## Installation

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to use

The simplest way to use this repository is:

1. Install the packages listed in `requirements.txt`
2. Open and run `main.ipynb`
3. Change the selected input file in the model-selection block
4. Rerun the notebook cells
5. Review the generated tables and plots

A typical model-selection block in `main.ipynb` looks like this:

```python
# Validation Truss
# model_path = "inputs/validation_truss.json"

# Validation Frame
# model_path = "inputs/validation_frame_original.json"
# model_path = "inputs/validation_frame_moment_release.json"
# model_path = "inputs/validation_frame_prescribed_displacement.json"
# model_path = "inputs/validation_frame_temperature_loads.json"
# model_path = "inputs/validation_frame_fabrication_error.json"

# Final Structure
model_path = "inputs/chaotianmen_bridge.json"
```

You can uncomment different files in the `inputs/` folder and run the notebook again to study different structural cases.

## Output control in `main.ipynb`

At the end of the workflow, `main.ipynb` mainly uses two postprocessing functions:

- `build_result_tables(...)`
- `plot_results(...)`

You do not need to change the solver code to use them. Instead, you should control their behavior through the display/save options depending on what you want.

### Table output options

Inside `build_result_tables(...)`, pay attention to:

- `display_tables=True/False`  
  Controls whether tables are shown directly in the notebook

- `save_tables=True/False`  
  Controls whether tables are exported to the `outputs/` folder

Example:
- Use `display_tables=True` if you want to inspect results immediately in the notebook
- Use `save_tables=True` if you want to create Excel outputs for later review

### Plot output options

Inside `plot_results(...)`, pay attention to:

- `display_3d=True/False`
- `save_3d=True/False`
- `display_2d=True/False`
- `save_2d=True/False`

These options control whether 2D and 3D plots are displayed during execution and/or saved to the `outputs/` folder.

Also note:

- `split_by_y_plane_2d=True`  
  Useful for the final bridge model because it separates the two longitudinal planes in 2D and makes the plots easier to read

In practice, you should adjust these flags depending on whether you want quick inspection, saved figures, or both.

## Recommended workflow

For general use, the following workflow is recommended:

1. Install the requirements
2. Open `main.ipynb`
3. Select one of the JSON files in `inputs/`
4. Run all cells
5. Check the displayed plots and tables
6. If needed, turn on or off the `display_*` and `save_*` options in:
   - `build_result_tables(...)`
   - `plot_results(...)`
7. Review saved files in `outputs/`