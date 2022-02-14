import sys
sys.path.insert(0, "../src")
import ase.io
from ase.visualize import view
import new_generate_crystals as cr

#file = "../input_structures/test/20_0.22_-8.02_400.cif"
file = "../input_structures/test/-1_3.58_-63.23_1564.cif"
structure = ase.io.read(file)
# view(structure)
n_atoms = 62
clash = cr.check_clash(structure, n_atoms, pbc=True, clash_type="inter", factor=1.5)
print(clash)
