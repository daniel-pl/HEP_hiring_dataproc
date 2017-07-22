""" Extracting author affiliations and other metadata from HEP publication records. """

import gzip
import os

import lxml.etree as ET
import pandas as pd

from utils.record_utils import *

AFFIL_NCOLS = 5
AFFIL_COLS = ["affiliation" + str(x) for x in range(1, AFFIL_NCOLS + 1)]
AFFIL_ID_COLS = ["affiliation" + str(x) + "ID" for x in range(1, AFFIL_NCOLS + 1)]

SUBJECT_NCOLS = 6
ARXIV_INSPIRE_SUBJECT_COLS = ["arXivSubject" + str(x) for x in range(1, SUBJECT_NCOLS + 1)] + \
                             ["INSPIRESubject" + str(x) for x in range(1, SUBJECT_NCOLS + 1)]


def get_author_aff(author_node):
    """ Get author's affiliations from a publication record. """
    author = get_subfield(author_node, "a")
    author_id = get_subfield(author_node, "x")
    author_id_curated = get_subfield(author_node, "y")

    affils = get_subfield(author_node, "u", take_first=False)
    affils_id = get_subfield(author_node, "z", take_first=False)

    affils = (affils + [np.nan] * AFFIL_NCOLS)[:AFFIL_NCOLS]
    affils_id = (affils_id + [np.nan] * AFFIL_NCOLS)[:AFFIL_NCOLS]

    author_df = pd.DataFrame({'author': [author], 'authorID': [author_id], 'authorIDcurated': [author_id_curated]})
    author_df = author_df.join(pd.DataFrame(data=[affils + affils_id], columns=AFFIL_COLS + AFFIL_ID_COLS))

    return author_df


def get_subjects(subject_nodes):
    """ Get subject areas of a publication record. """
    arXiv_subjects = []
    INSPIRE_subjects = []

    for subject in subject_nodes:
        category = get_subfield(subject, "2")
        subject = get_subfield(subject, "a")

        if category == "arXiv":
            arXiv_subjects.append(subject)
        elif category == "INSPIRE":
            INSPIRE_subjects.append(subject)

    arXiv_subjects = (arXiv_subjects + [np.nan] * SUBJECT_NCOLS)[:SUBJECT_NCOLS]
    INSPIRE_subjects = (INSPIRE_subjects + [np.nan] * SUBJECT_NCOLS)[:SUBJECT_NCOLS]

    subjects_df = pd.DataFrame(data=[arXiv_subjects + INSPIRE_subjects], columns=ARXIV_INSPIRE_SUBJECT_COLS)

    return subjects_df


def make_author_aff_record_df(record, file_out=""):
    """ Build dataframe of author's affiliations and other metadata from a single publication record. """
    recid = get_controlfield(record, "001")
    date = get_ds_field(record, "961", "c")
    date_preprint = get_ds_field(record, "269", "c")

    arXiv_subject_primary = get_ds_field(record, "037", "c")

    subject_nodes = record.findall("datafield[@tag='650']")
    subjects = get_subjects(subject_nodes)

    first_author_node = record.find("datafield[@tag='100']")
    other_authors_nodes = record.findall("datafield[@tag='700']")

    record_df = []
    if first_author_node is not None:
        record_df.append(get_author_aff(first_author_node))
    for author_node in other_authors_nodes:
        record_df.append(get_author_aff(author_node))

    if len(record_df) == 0:
        return

    record_df = pd.concat(record_df, axis=0, ignore_index=True)

    record_df = record_df.join(pd.concat([subjects] * len(record_df.index), ignore_index=True))

    record_df['arXivSubjectPrimary'] = arXiv_subject_primary
    record_df['recid'] = recid
    record_df['date'] = date
    record_df['datePreprint'] = date_preprint

    cols = record_df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    record_df = record_df[cols]

    if file_out:
        if not os.path.isfile(file_out):
            record_df.to_csv(file_out, header=cols, index=False, encoding='utf-8')
        else:
            record_df.to_csv(file_out, mode='a', header=False, index=False, encoding='utf-8')

    else:
        return record_df


def make_author_aff(file_in, file_out="", verbose=True):
    """ Build dataframe of all authors' affiliations and other metadata. """
    print("Processing records in " + file_in + " (this can take a long time)")
    author_aff_df = []
    context = ET.iterparse(gzip.GzipFile(file_in), tag='record')

    fast_iter(context, verbose, file_out, make_author_aff_record_df, author_aff_df)

    if len(author_aff_df) > 0:
        author_aff_df = pd.concat(author_aff_df, axis=0, ignore_index=True)

    print("\n Done.")
    return author_aff_df
