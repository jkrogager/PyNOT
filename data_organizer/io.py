# -*- coding: UTF-8 -*-

"""
Input / Output functions for the DataSet class.
"""

from data_organizer import TagDatabase


def save_database(database, output_fname):
    """Save file database to file."""
    collection = database.file_database
    output_strings = ['%s: %s' % item for item in collection.items()]
    # Sort the files based on their classification:
    sorted_output = sorted(output_strings, key=lambda x: x.split(':')[1])
    with open(output_fname, 'w') as output:
        output.write("\n".join(sorted_output))


def load_database(input_fname):
    """Load file database from file."""
    with open(input_fname) as input_file:
        all_lines = input_file.readlines()

    file_database = {}
    for line in all_lines:
        fname, ftype = line.split(':')
        file_database[fname.strip()] = ftype.strip()

    return TagDatabase(file_database)
