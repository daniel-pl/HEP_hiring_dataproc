""" Helper functions for record processing. """

import numpy as np
import sys


def get_controlfield(record, tag):
    """ Read a controlfield node with given tag inside a record. """
    path = "controlfield[@tag='" + tag + "']"
    try:
        controlfield = record.find(path).text
    except AttributeError:
        controlfield = np.nan
    return controlfield


def get_ds_field(record, tag, code, take_first=True):
    """ Read a datafield/subfield node with given tag and code inside a record.
    If take_first is True, only first subfield with given code is
    returned.
    """
    path = "datafield[@tag='" + tag + "']" + "/subfield[@code='" + code + "']"

    if take_first:
        try:
            ds_field = record.find(path).text
        except AttributeError:
            ds_field = np.nan

        return ds_field

    else:
        ds_field = []
        for element in record.findall(path):
            try:
                ds_field.append(element.text)
            except AttributeError:
                ds_field.append(np.nan)

        return ds_field


def get_subfield(datafield, code, take_first=True):
    """ Get content of subfield nodes with given code inside a datafield.
    If take_first is True, only first subfield with given code is
    returned.
    """
    path = "subfield[@code='" + code + "']"

    if take_first:
        try:
            subfield = datafield.find(path).text
        except AttributeError:
            subfield = np.nan

        return subfield

    else:
        subfield = []
        for element in datafield.findall(path):
            try:
                subfield.append(element.text)
            except AttributeError:
                subfield.append([])

        return subfield


def print_progress(counter, mark):
    if counter % (5 * mark) == 0:
        sys.stdout.write("%d" % counter)
        sys.stdout.flush()
    elif counter % mark == 0:
        sys.stdout.write(".")
        sys.stdout.flush()


def fast_iter(context, verbose, file_out, make_record_df, record_df):
    """ Iteratively process records in parsed XML stored in context. The
    processing function should be provided in the make_record_df argument.
    The processed record is appended to the record_df data frame.
    """
    counter = 0
    if verbose:
        print("Number processed:")

    for event, record in context:
        counter += 1
        if verbose:
            print_progress(counter, 200)

        if file_out:
            make_record_df(record, file_out=file_out)
        else:
            record_df.append(make_record_df(record))

        record.clear()
        while record.getprevious() is not None:
            del record.getparent()[0]

    del context
