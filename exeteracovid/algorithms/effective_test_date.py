import numpy as np


def effective_test_date(datastore, start_timestamp, end_timestamp,
                        created_at, date_taken_specific,
                        date_taken_between_start, date_taken_between_end,
                        dest_group, dest_name, filter_group, filter_name):
    """
    Filter tests for well-formedness and sensible date range specified.

    :param datastore: The Exetera session instance.
    :param start_timestamp: The earliest timestamp to filter on the tests.
    :param end_timestamp: The latest timestamp to filter on the tests.
    :param created_at: The 'create_at' column from Covid dataset.
    :param date_taken_specific: The 'date_taken_specific' column from Covid dataset.
    :param date_taken_between_start: The 'date_taken_between_start' column from Covid dataset.
    :param date_taken_between_end: The 'date_taken_between_end' column from Covid dataset.
    :param dest_group: The destination field to write the result to.
    :param dest_name: The name of the destination field.
    :param filter_group: The filter field to write the result filter to.
    :param filter_name: The name of the filter field.
    :return: A filtered set of effective test.
    """

    raw_t_cats = created_at[:]
    raw_t_dts = date_taken_specific[:]
    raw_t_dsbs = date_taken_between_start[:]
    raw_t_dsbe = date_taken_between_end[:]
    test_filter = np.zeros(len(raw_t_cats), dtype=np.bool)

    # remove tests where no dates are set
    cur_filter = np.logical_not((raw_t_dts == 0) & (raw_t_dsbs == 0) & (raw_t_dsbe == 0))
    test_filter = cur_filter[:]
    print("standard test filter 1:", np.count_nonzero(test_filter), len(test_filter))

    # remove tests where all three dates are set
    cur_filter = np.logical_not((raw_t_dts != 0) & (raw_t_dsbs != 0) & (raw_t_dsbe != 0))
    test_filter = test_filter & cur_filter
    print("standard test filter 2:", np.count_nonzero(test_filter), len(test_filter))

    # remove tests where only one of the date range tests is set
    cur_filter = np.logical_not((raw_t_dsbs != 0) & (raw_t_dsbe == 0) |
                                (raw_t_dsbs == 0) & (raw_t_dsbe != 0))
    test_filter = test_filter & cur_filter
    print("standard test filter 3:", np.count_nonzero(test_filter), len(test_filter))

    eff_test_dates = np.where(test_filter == False,
                              np.nan,
                              np.where(raw_t_dts != 0,
                                       raw_t_dts,
                                       raw_t_dsbs + (raw_t_dsbe - raw_t_dsbs) / 2))

    # remove tests where the test date is after the created at date
    cur_filter = eff_test_dates <= raw_t_cats
    test_filter = test_filter & cur_filter
    print("standard test filter 7:", np.count_nonzero(test_filter), len(test_filter))

    # remove tests with an effective test date outside permissible range
    cur_filter = (eff_test_dates >= start_timestamp) & (eff_test_dates <= end_timestamp)
    test_filter = test_filter & cur_filter
    print("standard_test_filter 8:", np.count_nonzero(test_filter), len(test_filter))

    print(len(eff_test_dates))
    print(len(test_filter))

    datastore.get_timestamp_writer(dest_group, dest_name,
                                   writemode='overwrite').write(eff_test_dates)
    datastore.get_numeric_writer(filter_group, filter_name, 'bool',
                                 writemode='overwrite').write(test_filter)

    return eff_test_dates, test_filter
