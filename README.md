
# NOTE: This code is deprecated and has been included in the `crystal` module of the [`MolDS` package](https://github.com/manucordova/MolDS)

# NMRX

NMRX stands for NMR Crystallography. This project uses experimental chemical shifts to drive structure generation schemes towards reasonable crystal structures. This is done through Monte-Carlo simulated annealing. Please read the [paper describing the method](https://dx.doi.org/10.1021/jacs.1c13733).

## Requirements

- Python 3 (3.7)
  - Numpy (1.19.1)
  - ASE (3.19.1)
  - [QUIP/quippy](http://libatoms.github.io/QUIP/) [will soon be updated with librascal]
- [DFTB+](https://dftbplus.org)
- [ShiftML kernels](https://www.materialscloud.org/work/tools/shiftml) [will soon be updated with ShiftML2]

## Subdirectories

- input_structures: Contains the initial conformer for each molecule explored. Any random conformer can be used.
- src: Contains source files for the simulations.
- run: Contain run script and SLURM submission scripts to run the experiments on a cluster.
