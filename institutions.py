""" Processing of Institutions records."""

import gzip
import os

import lxml.etree as ET
import pandas as pd

from utils.record_utils import *


def get_geo_coords(record):
    """ Get institution's geographic coordinates. """
    longitude = get_ds_field(record, "034", "d")
    latitude = get_ds_field(record, "034", "f")

    if pd.isnull(latitude) and not pd.isnull(longitude):
        latitude = get_ds_field(record, "034", "d", take_first=False)[1]

    return longitude, latitude


def make_instit_record_df(record, file_out=""):
    """ Process individual institution records. """
    recid = get_controlfield(record, "001")

    instit_native = get_ds_field(record, "110", "a")
    dept_native = get_ds_field(record, "110", "b")
    instit_HEP_new = get_ds_field(record, "110", "t")
    instit_HEP_old = get_ds_field(record, "110", "u")

    longitude, latitude = get_geo_coords(record)

    address_node = record.find("datafield[@tag='371']")

    if address_node:
        city = get_subfield(address_node, 'b')
        country = get_subfield(address_node, 'd')
        ZIP_code = get_subfield(address_node, 'e')
        country_code = get_subfield(address_node, 'g')
    else:
        city, country, ZIP_code, country_code = [np.nan] * 4

    instit_record_df = pd.DataFrame({'recid': [recid], 'InstituteNameNative': [instit_native],
                                     'DepartmentNameNative': [dept_native], 'InstituteNameHEPnew': [instit_HEP_new],
                                     'InstituteNameHEPold': [instit_HEP_old], 'City': [city], 'ZIPcode': [ZIP_code],
                                     'Country': [country], 'CountryCode': [country_code], 'Latitude': [latitude],
                                     'Longitude': [longitude]
                                     })

    cols = instit_record_df.columns.tolist()

    if file_out:
        if not os.path.isfile(file_out):
            instit_record_df.to_csv(file_out, header=cols, index=False, encoding='utf-8')
        else:
            instit_record_df.to_csv(file_out, mode='a', header=False, index=False, encoding='utf-8')

    else:
        return instit_record_df


def make_instits(file_in, file_out="", verbose=True):
    """ Build dataframe of processed institution records. """
    print("Processing records in " + file_in)
    instit_df = []
    context = ET.iterparse(gzip.GzipFile(file_in), tag='record')

    fast_iter(context, verbose, file_out, make_instit_record_df, instit_df)

    if len(instit_df) > 0:
        instit_df = pd.concat(instit_df, axis=0, ignore_index=True)

    print("\n Done.")
    return instit_df
