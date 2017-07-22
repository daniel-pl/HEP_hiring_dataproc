""" Processing of HepNames records. """

import gzip
import os

import lxml.etree as ET
import pandas as pd

from utils.record_utils import *

ALIAS_NCOLS = 5
ALIAS_COLS = ["alias" + str(x) for x in range(1, ALIAS_NCOLS + 1)]


def surname_initial(name):
    """ Transform name to 'Surname, First initial' form. """
    if pd.isnull(name):
        return "", ""

    name_split = name.split(",")

    try:
        surname = name_split[0]
    except IndexError:
        surname = ""

    try:
        first_name = name_split[1].split()[0]
        initial = first_name[0] + "."
    except IndexError:
        initial = ""

    return surname, initial


def make_author_record_df(record, file_out=""):
    """ Process individual author records.  """

    recid = get_controlfield(record, "001")

    name_last_first = get_ds_field(record, "100", "a")
    name_first_last = get_ds_field(record, "100", "q")

    name_last, initial_first = surname_initial(name_last_first)
    name_last_initial_first = name_last + ', ' + initial_first

    name_alias = get_ds_field(record, "400", "a", take_first=False)
    name_alias = (name_alias + [""] * ALIAS_NCOLS)[:ALIAS_NCOLS]

    author_record_df = pd.DataFrame(
        {'recid': [recid], 'NameLastFirst': [name_last_first], 'NameFirstLast': [name_first_last],
         'NameLast': [name_last], 'NameInitials': [initial_first], 'NameLastInitialFirst': [name_last_initial_first]})

    author_record_df = author_record_df.join(pd.DataFrame(data=[name_alias], columns=ALIAS_COLS))

    cols = author_record_df.columns.tolist()

    if file_out:
        if not os.path.isfile(file_out):
            author_record_df.to_csv(file_out, header=cols, index=False, encoding='utf-8')
        else:
            author_record_df.to_csv(file_out, mode='a', header=False, index=False, encoding='utf-8')

    else:
        return author_record_df


def make_authors(file_in, file_out="", verbose=True):
    """ Build dataframe of processed author records. """
    print("Processing records in " + file_in)
    authors_df = []
    context = ET.iterparse(gzip.GzipFile(file_in), tag='record')

    fast_iter(context, verbose, file_out, make_author_record_df, authors_df)

    if len(authors_df) > 0:
        authors_df = pd.concat(authors_df, axis=0, ignore_index=True)

    print("\n Done.")
    return authors_df
