# from PyNOT import *
# import extraction
# from extraction import *
# import alfosc
# import calibs
# from sens import sensitivity
import os

code_dir = os.path.dirname(os.path.abspath(__file__))
v_file = os.path.join(code_dir, 'VERSION')
with open(v_file) as version_file:
    __version__ = version_file.read().strip()

print("")
print("   PyNOT  v. %r " % __version__)
print("")
print("   Data Reduction Tools for ALFOSC")
print("   Mounted at the Nordic Optical Telescope")
print("")
print("   Written by Jens-Kristian Krogager")
print("   Institut d'Astrophysique de Paris")
print("   March 2017")
print("")
