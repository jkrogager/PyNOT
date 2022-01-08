import os
import yaml

code_dir = os.path.dirname(os.path.abspath(__file__))
cfg_fname = os.path.join(code_dir, '.instrument.cfg')

if os.path.exists(cfg_fname):
    with open(cfg_fname, 'r') as setup:
        cfg = yaml.full_load(setup)
    current_instrument = os.path.join(code_dir, cfg['current_instrument'])

else:
    print(' [WARNING] - Could not locate setup file %s in the PyNOT directory' % cfg_fname)
    print('             Using default : INSTRUMENT = ALFOSC')
    print('             If you want to use another instrument, you should run:')
    print('                 pynot setup')
    print('             or to see the list of installed instruments, run:')
    print('                 pynot use --list')
    print('')
    alfosc_path = os.path.join(code_dir, 'alfosc.py')
    current_instrument = alfosc_path

# -- TODO:
# include a test that `cfg` is proper, i.e. that it contains the key: 'instrument'

import importlib.util
spec = importlib.util.spec_from_file_location('instrument', current_instrument)
instrument = importlib.util.module_from_spec(spec)
spec.loader.exec_module(instrument)
