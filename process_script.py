""" Script processing records from INSPIRE dump, and building authors' timelines based on the records. """

from utils.author_aff_utils import *
from authors import make_authors
from institutions import make_instits
from author_aff import make_author_aff
from timelines import get_author_timeline_batch

import hashlib
import os
import sys
import urllib

data_url = "http://inspirehep.net/dumps/"
data_dir = "./INSPIREdump/"
clean_dir = "./records_clean/"

instits_filename = "Institutions-records.xml.gz"
authors_filename = "HepNames-records.xml.gz"
author_aff_filename = "HEP-records.xml.gz"

last_percent = None


def md5(filepath):
    """ Get md5 checksum of a file. """
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_progress(count, block_size, total_size):
    """ Report the download progress. """
    global last_percent
    percent = int(count * block_size * 100 / total_size)

    if last_percent != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent = percent


def download_file(filename):
    """ Download file from INSPIRE server. """
    print('Downloading file', filename)
    urllib.urlretrieve(data_url + filename, os.path.join(data_dir, filename), reporthook=download_progress)
    print('\n Download complete!')


def clean_name(filename):
    """ Rename file with the cleaned record. """
    return os.path.splitext(os.path.splitext(filename)[0])[0] + '_clean.csv'


def clean_final_name(filename):
    """ Rename file with the final cleaned record. """
    return os.path.splitext(os.path.splitext(filename)[0])[0] + '_clean_final.csv'


def check_update(filename):
    """ Check if a newer version of the file is available in the INSPIRE dump. """
    print("Checking for updates of " + filename + "...")
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        print('The file ' + filename + ' does not exist in the folder ' + data_dir + '.')
    else:
        inspire_md5 = urllib.urlopen(data_url + filename + ".md5").read()[:-1]
        file_md5 = md5(filepath)
        if file_md5 != inspire_md5:
            print('There is a newer version of the file', filename, 'on the INSPIRE server.')
        else:
            print("Done.")
            return

    response = raw_input("Do you wish to download the file from the INSPIRE server? [y/n]").lower()
    if response == "y":
        download_file(filename)

    print("Done.")


check_update(instits_filename)
check_update(authors_filename)
check_update(author_aff_filename)

make_instits(file_in=os.path.join(data_dir, instits_filename),
             file_out=os.path.join(clean_dir, clean_name(instits_filename)))

make_authors(file_in=os.path.join(data_dir, authors_filename),
             file_out=os.path.join(clean_dir, clean_name(authors_filename)))

# The processing of the entire file can take several hours on a standard laptop.
make_author_aff(file_in=os.path.join(data_dir, author_aff_filename),
                file_out=os.path.join(clean_dir, clean_name(author_aff_filename)))

print("Postprocessing records...")

print("Reading cleaned records")
dateparse = lambda x: pd.to_datetime(x, errors='coerce')
aff_records = pd.read_csv(os.path.join(clean_dir, clean_name(author_aff_filename)),
                          parse_dates=['date', 'datePreprint'], date_parser=dateparse)
# Here we provide the institution records already augmented with the city geolocation data. These were obtained
# with the 'get_city_geoloc' function, which uses the Google Maps API.
instit_records = pd.read_csv(os.path.join(clean_dir, clean_final_name(instits_filename)))

print("Removing records with missing affiliation or author ID")
aff_records = aff_records[aff_records['affiliation1'].notnull() & aff_records['authorID'].notnull()]
aff_records = adjust_date(aff_records)

print("Merging with institution data")
aff_records = merge_aff_instit(aff_records, instit_records)

print("Sorting by date and author ID")
aff_records.sort_values(['authorID', 'dateFirst'], inplace=True)

print("Writing to file")
aff_records.to_csv(os.path.join(clean_dir, clean_final_name(author_aff_filename)), index=False, encoding='utf-8')

print("Done.")

dateparse = lambda x: pd.to_datetime(x, errors='coerce')
aff_records = pd.read_csv(os.path.join(clean_dir, clean_final_name(author_aff_filename)),
                          parse_dates=['date', 'datePreprint', 'dateFirst'], date_parser=dateparse)

get_author_timeline_batch(aff_records, file_out=os.path.join(clean_dir, "author_timelines.csv"))
