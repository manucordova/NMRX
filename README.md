# NMRX
NMRX stands for NMR Crystallography. This uses experimental chemical shifts to drive structure generation schemes towards reasonable crystal structures.

# TODO

## Structure generation
- Restrict the cell volume to a (fixed) factor times the volume of the single-molecule.
- Change the way rotation steps are generated in order to make the rotations uniform.
- Get the third cell length value after determining the cell angles and other cell lengths, in order to have a reasonable volume for the starting crystal.
- Weight trial dihedral angles with probabilities determined with Mogul
- Generate cell lengths after cell angles and orientation of the molecule, according to the length of the molecule projected onto the corresponding direction

## Monte-Carlo run
- Change the sampling from Gaussian to uniform
- Change the initial temperature (2.5kJ, even 25kJ ?)
- Maybe do basement hopping to search for different minima ?
- Induce a change in volume when changing the orientation of the molecule in order to make clashes less likely

## Misc
- Restrict the spacegroup for chiral molecules to spacegroup without inversion centers or mirror planes
- Change ASE crystals with CCDC objects to avoid wrapping issues
