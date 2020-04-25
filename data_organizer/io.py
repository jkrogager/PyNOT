# -*- coding: UTF-8 -*-

"""
Input / Output functions for the DataSet class.
"""

from DataOrganizer import DataSet, get_pipeline_version


def save_dataset(dataset, output_fname, verbose=True):
    """Save DataSet to file."""
    with open(output_fname, 'w') as out:
        # -- Write metadata
        out.write("# X-shooter dataset\n")
        out.write("version = %s\n" % dataset.version)
        # -- Write file-database:
        for key, values in dataset.tag_database.items():
            for fname in values:
                out.write("%s : %s\n" % (fname, key))

    if verbose:
        print(" Saved dataset to file: %s\n" % output_fname)


def load_dataset(input_fname):
    """Load Dataset from file."""
    with open(input_fname) as input_file:
        all_lines = input_file.readlines()

    file_database = dict()
    for line in all_lines:
        if line[0] == '#':
            continue
        elif 'version = ' in line:
            version = line.split('=')[1].strip()
        else:
            fname, category = line.split(' : ')
            file_database[fname.strip()] = category.strip()

    current_version = get_pipeline_version()
    if current_version != version:
        print("  [WARNING] - Pipeline version in dataset does not match installed version of esorex!")
        print("              Dataset: %s    ESOREX: %s" % (version, current_version))
        print("              Consider regenerating the dataset from scratch.")

    # -- Initiate DataSet:
    xsh_dataset = DataSet(set_static=False)
    xsh_dataset.set_tag_database(file_database)
    return xsh_dataset
