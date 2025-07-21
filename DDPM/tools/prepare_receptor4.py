#!/usr/bin/env python3
# Converted from Python 2 to Python 3

import os
import sys
import getopt
from MolKit import Read
import MolKit.molecule
import MolKit.protein
from AutoDockTools.MoleculePreparation import AD4ReceptorPreparation

def usage():
    print("Usage: prepare_receptor4.py -r filename")
    print("\n    Description of command...")
    print("         -r   receptor_filename ")
    print("        supported file types include pdb,mol2,pdbq,pdbqs,pdbqt, possibly pqr,cif")
    print("    Optional parameters:")
    print("        [-v]  verbose output (default is minimal output)")
    print("        [-o pdbqt_filename]  (default is 'molecule_name.pdbqt')")
    print("        [-A]  type(s) of repairs to make: ")
    print("             'bonds_hydrogens': build bonds and add hydrogens ")
    print("             'bonds': build a single bond from each atom with no bonds to its closest neighbor")
    print("             'hydrogens': add hydrogens")
    print("             'checkhydrogens': add hydrogens only if there are none already")
    print("             'None': do not make any repairs ")
    print("             (default is 'None')")
    print("        [-C]  preserve all input charges ie do not add new charges ")
    print("             (default is addition of gasteiger charges)")
    print("        [-p]  preserve input charges on specific atom types, eg -p Zn -p Fe")
    print("        [-U]  cleanup type:")
    print("             'nphs': merge charges and remove non-polar hydrogens")
    print("             'lps': merge charges and remove lone pairs")
    print("             'waters': remove water residues")
    print("             'nonstdres': remove chains composed entirely of residues of")
    print("                      types other than the standard 20 amino acids")
    print("             'deleteAltB': remove XX@B atoms and rename XX@A atoms->XX")
    print("             (default is 'nphs_lps_waters_nonstdres') ")
    print("        [-e]  delete every nonstd residue from any chain")
    print("        [-M]  interactive ")
    print("        [-d dictionary_filename] file to contain receptor summary information")
    print("        [-w]   assign each receptor atom a unique name: newname is original name plus its index(1-based)")

try:
    opt_list, args = getopt.getopt(sys.argv[1:], 'r:vo:A:Cp:U:eM:d:wh')
except getopt.GetoptError as msg:
    print(f'prepare_receptor4.py: {msg}')
    usage()
    sys.exit(2)

receptor_filename = None
verbose = None
repairs = ''
charges_to_add = 'gasteiger'
preserve_charge_types = None
cleanup = "nphs_lps_waters_nonstdres"
outputfilename = None
mode = 'automatic'
delete_single_nonstd_residues = None
dictionary = None
unique_atom_names = False

for o, a in opt_list:
    if o == '-r':
        receptor_filename = a
    elif o == '-v':
        verbose = True
    elif o == '-o':
        outputfilename = a
    elif o == '-A':
        repairs = a
    elif o == '-C':
        charges_to_add = None
    elif o == '-p':
        preserve_charge_types = a if not preserve_charge_types else preserve_charge_types + ',' + a
    elif o == '-U':
        cleanup = a
    elif o == '-e':
        delete_single_nonstd_residues = True
    elif o == '-M':
        mode = a
    elif o == '-d':
        dictionary = a
    elif o == '-w':
        unique_atom_names = True
    elif o == '-h':
        usage()
        sys.exit()

if not receptor_filename:
    print('prepare_receptor4: receptor filename must be specified.')
    usage()
    sys.exit()

mols = Read(receptor_filename)
if verbose:
    print(f'read {receptor_filename}')

mol = mols[0]
if unique_atom_names:
    for at in mol.allAtoms:
        if mol.allAtoms.get(at.name) > 1:
            at.name = at.name + str(at._uniqIndex + 1)
    if verbose:
        print(f"renamed {len(mol.allAtoms)} atoms")

preserved = {}
has_autodock_element = False
if charges_to_add is not None and preserve_charge_types is not None:
    if hasattr(mol, 'allAtoms') and not hasattr(mol.allAtoms[0], 'autodock_element'):
        _, file_ext = os.path.splitext(receptor_filename)
        if file_ext == '.pdbqt':
            has_autodock_element = True
    if not has_autodock_element:
        print('prepare_receptor4: cannot preserve charges due to missing autodock_element')
        sys.exit()

    preserved_types = preserve_charge_types.split(',')
    for t in preserved_types:
        if not len(t): continue
        ats = mol.allAtoms.get(lambda x: x.autodock_element == t)
        for a in ats:
            if a.chargeSet is not None:
                preserved[a] = [a.chargeSet, a.charge]

if len(mols) > 1:
    for m in mols[1:]:
        if len(m.allAtoms) > len(mol.allAtoms):
            mol = m

mol.buildBondsByDistance()
alt_loc_ats = mol.allAtoms.get(lambda x: "@" in x.name)
if len(alt_loc_ats):
    print(f"WARNING! {mol.name} has {len(alt_loc_ats)} alternate location atoms!")

RPO = AD4ReceptorPreparation(mol, mode, repairs, charges_to_add,
                             cleanup, outputfilename=outputfilename,
                             preserved=preserved,
                             delete_single_nonstd_residues=delete_single_nonstd_residues,
                             dict=dictionary)

if charges_to_add is not None:
    for atom, chargeList in preserved.items():
        atom._charges[chargeList[0]] = chargeList[1]
        atom.chargeSet = chargeList[0]
