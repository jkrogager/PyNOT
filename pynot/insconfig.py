"""
Script to install a new instrument configuration

Files needed (example given for alfosc):
  instrument configuration module : alfosc.py
  classification rule file        : data/alfosc.rules
  extinction data file            : calib/lapalma.ext
  filter definition table         : calib/alfosc_filters.dat
"""

import os
import importlib.util
import glob
import yaml

code_dir = os.path.dirname(os.path.abspath(__file__))
cfg_fname = os.path.join(code_dir, '.instrument.cfg')

mandatory_attributes = [
    'create_pixel_array',
    'extinction_fname',
    'filter_table_fname',
    'filter_translate',
    'get_airmass',
    'get_binning',
    'get_binning_from_hdr',
    'get_date',
    'get_exptime',
    'get_filter',
    'get_grism',
    'get_header',
    'get_header_info',
    'get_object',
    'get_rotpos',
    'get_slit',
    'get_target_name',
    'grism_translate',
    'name',
    'rulefile',
    'slits',
]

mandatory_filetags = ['BIAS', 'DARK', 'SPEC_FLAT', 'SPEC_OBJECT', 'ACQ_IMG', 'IMG_OBJECT', 'IMG_FLAT']

class RuleNotFound(Exception):
    pass

class RuleParsingError(Exception):
    pass

class TableColumnError(Exception):
    pass


def get_instrument_name(filename):
    pattern = "# pynot-instrument-module"
    with open(filename, 'r') as module:
        if pattern in module.readline():
            pycode = module.readlines()
            for line in pycode:
                if line.find('name =') == 0 or line.find('name=') == 0:
                    name = line.split('=')[1].strip()
                    name = name.replace('"', '').replace("'", "")
                    break
            else:
                name = ''
        else:
            name = ''
    return name


def get_installed_instruments():
    pattern = "# pynot-instrument-module"
    all_instruments = dict()
    all_files = glob.glob(os.path.join(code_dir, '*.py'))
    for fname in all_files:
        with open(fname, 'r') as mod:
            if pattern in mod.readline():
                # find the 'name' of the instrument
                pycode = mod.readlines()
                for line in pycode:
                    if line.find('name =') == 0 or line.find('name=') == 0:
                        name = line.split('=')[1].strip()
                        name = name.replace('"', '').replace("'", "")
                        break
                all_instruments[name] = fname

    max_name_length = max([len(key) for key in all_instruments.keys()])
    msg = list("\n")
    msg.append("  Following instruments are installed:")
    msg.append(("    %%%is  :  python file" % max_name_length) % 'name')
    for name, module_file in all_instruments.items():
        msg.append(("  - %%%is  :  %%s" % max_name_length) % (name, module_file))
    msg.append("")
    msg.append("  To switch to any of them run:")
    msg.append("    pynot use name")
    msg.append("")
    output_msg = "\n".join(msg)
    return all_instruments, output_msg


def make_default_configuration(current):
    cfg = {'current_instrument': os.path.basename(current)}
    instruments, _ = get_installed_instruments()
    all_instruments = {}
    for name, module_file in instruments.items():
        all_instruments[name] = os.path.basena(module_file)
    cfg['all_instruments'] = all_instruments
    return cfg


def update_instrument_cfg(current=None, all_instruments=None):
    """
    Update either the `current_instrument` or the list of `all_instruments`
    in the instrument configuration file.
    """
    if current is None and all_instruments is None:
        return "          - instrument configuration is up to date\n"

    msg = list()
    if os.path.exists(cfg_fname):
        with open(cfg_fname, 'r') as setup:
            cfg = yaml.full_load(setup)
        msg.append("          - opened instrument configuration file: %s" % cfg_fname)
    else:
        cfg = make_default_configuration('alfosc.py')
        msg.append("[WARNING] - setting the instrument configuration to default: alfosc.py")

    if current is not None:
        cfg['current_instrument'] = os.path.basename(current)
        msg.append("          - updated current instrument: %s" % os.path.basename(current))

    if all_instruments:
        cfg['all_instruments'] = all_instruments
        msg.append("          - updated instrument list: %r" % list(all_instruments.keys()))

    with open(cfg_fname, 'w') as setup:
        yaml.dump(cfg, setup)
        msg.append("          - saved instrument configuration file: %s" % cfg_fname)
    msg.append("")
    return "\n".join(msg)


def change_instrument(name, all_instruments):
    if not name:
        return

    if name not in all_instruments:
        return " [ERROR]  - the given instrument is not installed: %s" % name
    else:
        update_instrument_cfg(current=all_instruments[name])


def test_instrument_module(mod, filepath):
    """Check if all necessary attributes are defined in the module, or raise an AttributeError"""
    missing_attributes = list()
    for attr in mandatory_attributes:
        if not hasattr(mod, attr):
            missing_attributes.append(attr)

    if len(missing_attributes) > 0:
        missing_str = ', '.join(["'%s'" % a for a in missing_attributes])
        if len(missing_attributes) == 1:
            err_msg = f"module {filepath} has no attribute {missing_str}"
        else:
            err_msg = f"module {filepath} has no attributes: {missing_str}"
        raise AttributeError(err_msg)


def test_instrument_rules(rule_file):
    with open(rule_file) as rulebook:
        rules = rulebook.readlines()
    filetags = list()
    for linenum, rule in enumerate(rules, 1):
        if rule[0] == '#' or len(rule.strip()) == 0:
            continue

        elements = rule.split(':')
        if len(elements) == 2:
            ftag, criteria = elements
            filetags.append(ftag.strip())
        else:
            raise RuleParsingError("incorrect rule format in line %i: %s" % (linenum, rule))

    # filetags = [r.split(':')[0].strip() for r in rules]
    has_tags = [ftag in filetags for ftag in mandatory_filetags]
    if not all(has_tags):
        missing_tags = [ftag for (good, ftag) in zip(has_tags, mandatory_filetags) if not good]
        missing_str = ', '.join(["'%s'" % t for t in missing_tags])
        if len(missing_tags) == 1:
            err_msg = f"classification rule missing for type: {missing_str}"
        else:
            err_msg = f"classification rules missing for types: {missing_str}"
        raise RuleNotFound(err_msg)

    for ftag in filetags:
        if 'ARC' in ftag:
            has_arc = True
            break
    else:
        has_arc = False

    if not has_arc:
        raise RuleNotFound("classification rule missing for type: ARC")


def test_filter_table(filter_file):
    from astropy.table import Table
    filter_table = Table.read(filter_file, format='ascii.fixed_width')
    has_name = 'name' in filter_table.colnames
    has_short_name = 'short_name' in filter_table.colnames

    if has_name and has_short_name:
        pass
    else:
        missing_list = []
        if not has_name:
            missing_list.append("'name'")
        if not has_short_name:
            missing_list.append("'short_name'")
        missing_str = ', '.join(missing_list)

        if len(missing_list) == 1:
            err_msg = f"filter definition table has no column: {missing_str}"
        else:
            err_msg = f"filter definition table has no columns: {missing_str}"
        raise TableColumnError(err_msg)


# -- Entry point:
def setup_instrument(args):

    # -- Load the instrument configurations
    if os.path.exists(cfg_fname):
        with open(cfg_fname, 'r') as setup:
            cfg = yaml.full_load(setup)
        print("          - opened instrument configuration file: %s" % cfg_fname)
    else:
        cfg = make_default_configuration('alfosc.py')
        print("[WARNING] - something seems to be wrong with the instrument configuration file")
        print("[WARNING] - reinitializing the instrument configuration to the default settings")

    ins_path = os.path.abspath(args.module)
    name = get_instrument_name(ins_path)
    if name in cfg['all_instruments']:
        print(f" [ERROR]  - an instrument with the name {name} has already been installed")
        print("")
        return ""

    if not args.rules:
        prompt_msg = 'Filename of file classification rules:\n(path to file)\n > '
        user_input = input(prompt_msg)
        rule_path = os.path.abspath(user_input)
    else:
        rule_path = os.path.abspath(args.rules)

    if not args.filters:
        prompt_msg = 'Filename of filter definition table:\n(path to file)\n > '
        user_input = input(prompt_msg)
        filter_path = os.path.abspath(user_input)
    else:
        filter_path = os.path.abspath(args.filters)

    if not args.ext:
        prompt_msg = 'Filename of observatory extinction table:\n(path to file or one of: lasilla, lapalma, paranal)\n > '
        user_input = input(prompt_msg)
        if user_input.lower() in ['lasilla', 'lapalma', 'paranal']:
            ext_path = os.path.join(code_dir, 'calib/%s.ext' % user_input.lower())
            ext_copy = False
        else:
            ext_path = os.path.abspath(user_input)
            ext_copy = True
    else:
        if args.ext.lower() in ['lasilla', 'lapalma', 'paranal']:
            ext_path = os.path.join(code_dir, 'calib/%s.ext' % args.ext.lower())
            ext_copy = False
        else:
            ext_path = os.path.abspath(args.ext)
            ext_copy = True

    # -- Test instrument classification rules
    try:
        test_instrument_rules(rule_path)
        print("          - successfully imported rulebook: %s" % rule_path)
    except (FileNotFoundError, RuleNotFound, RuleParsingError) as e:
        print(" [ERROR]  - error in rulebook: %s" % rule_path)
        print(" [ERROR]  - " + str(e))
        print("")
        return ""

    # -- Test instrument filter table
    try:
        test_filter_table(filter_path)
        print("          - successfully imported filter table: %s" % filter_path)
    except (FileNotFoundError, TableColumnError, OSError) as e:
        print(" [ERROR]  - error in filter definition table: %s" % filter_path)
        print(" [ERROR]  - " + str(e))
        print("")
        return ""

    # -- Test if the observatory extinction file exists
    if os.path.exists(ext_path):
        print("          - successfully imported extinction data: %s" % ext_path)
    else:
        print(" [ERROR]  - The file does not exist: %s" % ext_path)
        print("")
        return ""

    # -- Test instrument module
    try:
        spec = importlib.util.spec_from_file_location('instrument', ins_path)
        instrument = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(instrument)

        test_instrument_module(instrument, ins_path)
        print("          - successfully imported module: %s" % ins_path)
    except AttributeError as e:
        print(" [ERROR]  - " + str(e))
        if "object has no attribute 'loader'" in str(e):
            print(" [ERROR]  - The given file does not seem to be a proper python module!")
        print("")
    except FileNotFoundError:
        pass
        # return ""


    # -- Copy files:
    print("")
    print("   This will install the following files into the PyNOT source directory:")
    print("    - %s" % ins_path)
    print("    - %s" % rule_path)
    print("    - %s" % filter_path)
    if ext_copy:
        print("    - %s" % ext_path)
    print("")

    answer = input("   Are you sure you want to continue?  (Y/n)\n > ")
    if not answer.lower() in ['y', 'yes', '']:
        print("  [ABORT] - Terminating the installation procedure!")
        print("")
        return ""

    calib_dir = os.path.join(code_dir, 'calib')
    data_dir = os.path.join(code_dir, 'data')
    if ext_copy:
        copy_ext_file = "cp %s %s" % (ext_path, calib_dir)
        os.system(copy_ext_file)

    copy_module = "cp %s %s" % (ins_path, code_dir)
    copy_filters = "cp %s %s" % (filter_path, calib_dir)
    copy_rules = "cp %s %s" % (rule_path, data_dir)
    os.system(copy_module)
    os.system(copy_filters)
    os.system(copy_rules)


    # -- Update the parameters of .instrument.cfg
    if args.use:
        cfg['current_instrument'] = instrument.name
        print("          - updated current instrument: %s" % instrument.name)

    cfg['all_instruments'][instrument.name] = os.path.basename(ins_path)

    with open(cfg_fname, 'w') as setup:
        yaml.dump(cfg, setup)
    print("          - saved instrument configuration file: %s" % cfg_fname)
    print("  [DONE]  - Successfully installed instrument: %s" % instrument.name)
    print("")
    return ""
