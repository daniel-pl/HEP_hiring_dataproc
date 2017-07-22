""" Postrpocessing of authors' affiliations and metadata dataframe. """

import pandas as pd

AFFIL_NCOLS = 5


def sort_aff(author_aff, file_out):
    """ Sort by author ID and first publication Date. """
    author_aff.sort_values(['authorID', 'dateFirst'], inplace=True)
    author_aff.to_csv(file_out, index=False, encoding='utf-8')

    return author_aff


def trunc_aff(author_aff, file_out):
    """ Remove rows with missing affiliation or author ID. """
    author_aff = author_aff[author_aff['affiliation1'].notnull() & author_aff['authorID'].notnull()]
    author_aff.to_csv(file_out, index=False, encoding='utf-8')

    return author_aff


def merge_aff_instit(author_aff, institutions):
    """ Merge with institution records to augment the affiliation information. """
    instit = institutions.drop(['DepartmentNameNative', 'InstituteNameNative', 'ZIPcode', 'Country'], axis=1)
    instit_with_ID = instit[instit['recid'].notnull()]
    instit_with_Name_old = instit[instit['InstituteNameHEPold'].notnull()]

    for i in range(1, AFFIL_NCOLS + 1):
        aff_column = 'affiliation' + str(i)
        affID_column = 'affiliation' + str(i) + 'ID'
        recid_column = 'recid' + str(i)
        instit_name_column = 'InstituteNameHEPold' + str(i)

        instit_with_ID.columns = [col + str(i) for col in instit.columns]
        instit_with_Name_old.columns = [col + str(i) for col in instit.columns]

        author_aff_with_ID = author_aff[author_aff[affID_column].notnull()]
        author_aff_no_ID = author_aff[author_aff[affID_column].isnull()]

        author_aff_with_ID = pd.merge(left=author_aff_with_ID, right=instit_with_ID, how='left', left_on=affID_column,
                                      right_on=recid_column, sort=False)
        author_aff_no_ID = pd.merge(left=author_aff_no_ID, right=instit_with_Name_old, how='left', left_on=aff_column,
                                    right_on=instit_name_column, sort=False)

        author_aff = pd.concat([author_aff_with_ID, author_aff_no_ID])

    return author_aff


def adjust_date(author_aff):
    """ Add column with first publication date. """
    author_aff['dateFirst'] = author_aff[['date', 'datePreprint']].min(axis=1)
    # author_aff = author_aff.drop(['date','datePreprint'], axis=1)
    return author_aff
