""" Determining researchers' career paths (timelines) from affiliation data (signatures). """

import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix
from sklearn import linear_model

import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

from utils.record_utils import print_progress

import os
import sys

AFFIL_NCOLS = 5
CLUSTER_RADIUS_X = 50.0
CLUSTER_RADIUS_T = 550.0
MIN_CLUSTER_SIZE = 2

counter = 0


def geo_dist(coord_df):
    """ Compute the geodesic distance between two geolocations using haversine formula. """
    coord = np.array(coord_df)
    coord = np.deg2rad(coord)

    lat = coord[:, 0]
    lng = coord[:, 1]

    diff_lat = lat[:, None] - lat
    diff_lng = lng[:, None] - lng

    d = np.sin(diff_lat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat) * np.sin(diff_lng / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(d))


def geo_dist_pair(coord_pair):
    """ Compute the geodesic distance between two geolocations using haversine formula. """
    coord1, coord2 = np.deg2rad(np.array(coord_pair[0:2])), np.deg2rad(np.array(coord_pair[2:4]))
    diff_lat, diff_lng = coord2 - coord1

    d = np.sin(diff_lat / 2) ** 2 + np.cos(coord1[0]) * np.cos(coord2[0]) * np.sin(diff_lng / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(d))


def date_dist(date_df):
    """ Compute the time difference in days between two dates. """
    time_dist = np.absolute((date_df[:, None] - date_df).astype('timedelta64[D]') / np.timedelta64(1, 'D'))
    return time_dist


def aff_dist(coord_instit, coord_city, date, spacetime_factor, use_INSPIRE_coord=False):
    """ Compute distance between affiliations. """
    date = np.array(date, dtype='datetime64')

    if use_INSPIRE_coord:
        space_dist = np.nanmean(np.array([geo_dist(coord_instit), geo_dist(coord_city)]), axis=0)
    else:
        space_dist = geo_dist(coord_city)

    time_dist = date_dist(date)

    dist = np.nanmax(np.array([space_dist, spacetime_factor * time_dist]), axis=0)

    return csr_matrix(dist)


def find_top(s):
    """ Find the most frequent entry. """
    try:
        return s.value_counts().index[0]
    except IndexError:
        return np.nan


def reduce_signatures(s):
    """ Drop unnecessary columns and put each affiliation in a separate row. """
    s_new = []
    s = s.drop(['date', 'datePreprint', 'author', 'authorID', 'authorIDcurated', 'recid'], axis=1)
    for row in s.itertuples():
        for i in range(AFFIL_NCOLS):
            s_new.append(list((row[1 + i],) + (row[24],) + row[25 + i * 9: 25 + (i + 1) * 9] + (i + 1,)))

    s_new = pd.DataFrame(s_new)
    s_new.columns = s.columns[[0] + [23] + range(24, 33)].append(pd.Index(['affiliationOrder']))
    s_new.columns = [c.replace('1', '') for c in s_new.columns]

    s_new = s_new[s_new['LatitudeCity'].notnull() & s_new['LongitudeCity'].notnull()]
    s_new = s_new.reset_index(drop=True)

    return s_new


def predict_timeline(signature_dates, labels, reg=1e20, plot_reg=False):
    """ Model author timeline using multinomial logistic regression. Features are publication dates, and labels are
    affiliation clusters. """
    dl = pd.DataFrame({'dateFirst': signature_dates, 'label': labels})
    dl = dl[(dl['label'] > -1)].reset_index(drop=True)

    if dl['dateFirst'].nunique() == 1:
        date_range = [dl.loc[0, 'dateFirst']] * 2
        yp = [dl.loc[0, 'label']] * 2
        timeline = pd.DataFrame({'date': date_range, 'label': yp})
        return timeline

    date_range = pd.date_range(min(dl['dateFirst']).replace(day=1), max(dl['dateFirst']).replace(day=1), freq='MS')

    if dl['label'].nunique() == 1:
        yp = [dl.loc[0, 'label']] * len(date_range)
        timeline = pd.DataFrame({'date': date_range, 'label': yp})
        return timeline

    X = dl['dateFirst'].apply(lambda x: x.toordinal()).values.reshape(-1, 1)
    mu_X = np.mean(X)
    std_X = np.std(X)
    X = (X - mu_X) / std_X
    y = np.array(dl['label'])

    logit_reg = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', C=reg)
    logit_reg.fit(X, y)

    Xp = map(lambda x: x.toordinal(), date_range)
    Xp = (Xp - mu_X) / std_X
    yp = map(logit_reg.predict, Xp)
    yp = [x[0] for x in yp]

    if plot_reg:
        plt.figure(1, figsize=(12, 8))
        plt.plot(dl['dateFirst'], y, "o", color='black', zorder=20)
        plt.plot(date_range, yp, color='red', linewidth=3)
        plt.show()

    timeline = pd.DataFrame({'date': date_range, 'label': yp})
    return timeline


def build_timeline(timeline_l):
    """ Make timeline indicating start and end date for each affiliation. """
    timeline_l['block'] = (timeline_l['label'].shift(1) != timeline_l['label']).astype(int).cumsum()

    B = timeline_l.groupby('block')
    aff_start_date = B['date'].apply(lambda x: min(x))
    aff_end_date = B['date'].apply(lambda x: max(x))
    aff_label = B['label'].apply(lambda x: x.iloc[0])

    timeline = pd.DataFrame({'affiliationStartDate': aff_start_date, 'affiliationEndDate': aff_end_date,
                             'label': aff_label})
    timeline = timeline[['affiliationStartDate', 'affiliationEndDate', 'label']].reset_index(drop=True)

    return timeline


def cluster_rep(cluster, aff_order_pref=False):
    """ Pick the most frequent institution as representative of a cluster. """
    if aff_order_pref:
        top_recid = find_top(cluster[cluster['affiliationOrder'] == 1]['recid'])
    else:
        top_recid = find_top(cluster['recid'])

    return cluster[cluster['recid'] == top_recid].iloc[0].drop('affiliationOrder')


def verify_consecutive(timeline, cluster_radius_x):
    """ Merge consecutive affiliations if within cluster radius from each other. """
    timeline[['LatitudeCityNext', 'LongitudeCityNext']] = timeline[['LatitudeCity', 'LongitudeCity']].shift(-1)
    timeline['distFromNext'] = timeline[['LatitudeCityNext', 'LongitudeCityNext',
                                         'LatitudeCity', 'LongitudeCity']].apply(geo_dist_pair, axis=1)

    first_start_date = timeline.loc[0, 'affiliationStartDate']

    timeline = timeline[(timeline['distFromNext'] > cluster_radius_x) | timeline['distFromNext'].isnull()]
    timeline = timeline.reset_index(drop=True)

    timeline.set_value(0, 'affiliationStartDate', first_start_date)

    timeline.loc[1:, 'affiliationStartDate'] = \
        timeline['affiliationEndDate'].shift(1).loc[1:].apply(lambda x: x + relativedelta(months=1))

    timeline = timeline.drop(['LatitudeCityNext', 'LongitudeCityNext', 'distFromNext'], axis=1).reset_index(drop=True)

    return timeline


def add_previous_next_cols(timeline):
    """ Add columns with next and previous affiliation. """
    timeline_aff = timeline.loc[:, 'City':]

    timeline_previous = timeline_aff.shift(1).add_suffix('Previous')
    timeline_next = timeline_aff.shift(-1).add_suffix('Next')

    timeline = pd.concat([timeline, timeline_previous, timeline_next], axis=1)

    return timeline


def add_author_info(timeline, author_name, authorID, top_arXiv_subject, top_INSPIRE_subject):
    """ Add columns with basic author information. """
    timeline['author'] = author_name
    timeline['authorID'] = authorID
    timeline['arXivSubjectTop'] = top_arXiv_subject
    timeline['INSPIRESubjectTop'] = top_INSPIRE_subject

    cols = timeline.columns.tolist()
    cols = cols[-4:] + cols[:-4]
    return timeline[cols]


def build_author_timeline(author_signatures, file_out, verbose=True,
                          cluster_radius_x=CLUSTER_RADIUS_X, cluster_radius_t=CLUSTER_RADIUS_T,
                          min_cluster_size=MIN_CLUSTER_SIZE,
                          print_clusters=False, plot_reg=False):
    """ Make author's timeline from signatures. """
    global counter
    top_arXiv_subject = find_top(author_signatures['arXivSubjectPrimary'])
    top_INSPIRE_subject = find_top(author_signatures['INSPIRESubject1'])
    author_name = find_top(author_signatures['author'])
    authorID = author_signatures['authorID'].iloc[0]

    # Expand affiliations and throw out some of the columns and rows with NaN geo coordinates.
    s = reduce_signatures(author_signatures)

    # Compute distance matrix.
    dist_matrix = aff_dist(s[['Latitude', 'Longitude']], s[['LatitudeCity', 'LongitudeCity']], s[['dateFirst']],
                           cluster_radius_x / cluster_radius_t, use_INSPIRE_coord=False)

    # If only one data point return empty dataframe.
    if dist_matrix.getnnz() == 0:
        return pd.DataFrame()

    # Find affiliation clusters and find their representatives.
    clustering = DBSCAN(eps=cluster_radius_x, min_samples=min_cluster_size, metric='precomputed').fit(dist_matrix)

    non_negative_labels = set([x for x in clustering.labels_ if x > -1])
    if len(non_negative_labels) == 0:
        return pd.DataFrame()

    labeled_instit = s.loc[:, 'City':]
    labeled_instit['label'] = clustering.labels_
    labeled_instit = labeled_instit[labeled_instit['label'] > -1]

    if print_clusters:
        pd.set_option('display.max_rows', len(labeled_instit))
        print(labeled_instit[['InstituteNameHEPold', 'label']].reset_index(drop=True))
        pd.reset_option('display.max_rows')

    clusters = labeled_instit.groupby('label').apply(cluster_rep)

    # Predict affiliation cluster for each record's date.
    timeline_label = predict_timeline(s['dateFirst'], clustering.labels_, plot_reg=plot_reg)

    # Show From-To dates for each label.
    author_timeline = build_timeline(timeline_label)

    # Merge labels with cluster representative information.
    author_timeline = pd.merge(left=author_timeline, right=clusters, on='label', how='left', right_index=True)
    author_timeline = verify_consecutive(author_timeline, cluster_radius_x)
    author_timeline = add_previous_next_cols(author_timeline)
    author_timeline = add_author_info(author_timeline, author_name, authorID, top_arXiv_subject, top_INSPIRE_subject)

    if os.path.isfile(file_out):
        author_timeline.to_csv(file_out, mode='a', index=False, encoding='utf-8', header=False)
    else:
        author_timeline.to_csv(file_out, mode='a', index=False, encoding='utf-8')

    counter += 1
    if verbose:
        print_progress(counter, 50)

    return author_timeline


def get_author_timeline_batch(signatures, file_out, verbose=True,
                              cluster_radius_x=CLUSTER_RADIUS_X, cluster_radius_t=CLUSTER_RADIUS_T,
                              min_cluster_size=MIN_CLUSTER_SIZE):
    """ Get timelines of all authors in a list of signatures. """
    print("Building author timelines... (this can take a long time)")
    signatures.groupby('authorID').apply(build_author_timeline, file_out, verbose,
                                         cluster_radius_x, cluster_radius_t, min_cluster_size)
    print("Done.")
