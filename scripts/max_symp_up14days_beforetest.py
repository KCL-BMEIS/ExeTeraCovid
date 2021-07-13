import itertools
import numpy as np
import pandas as pd
import exetera as hy
import h5py
import glob
from datetime import datetime, timezone

from exetera.core.persistence import DataStore
from exetera.core import utils
from exetera.core import persistence as prst
from exetera.processing.test_type_from_mechanism import test_type_from_mechanism_v1
from exetera.core import session

ds = DataStore()
ts = str(datetime.now(timezone.utc))

list_symptoms = ['fatigue', 'abdominal_pain', 'chest_pain', 'sore_throat', 'shortness_of_breath',
                 'skipped_meals', 'loss_of_smell', 'unusual_muscle_pains', 'headache', 'hoarse_voice', 'delirium',
                 'diarrhoea',
                 'fever', 'persistent_cough', 'dizzy_light_headed', 'eye_soreness', 'red_welts_on_face_or_lips',
                 'blisters_on_feet']

with session.Session() as s:
    source = s.open_dataset('/mnt/data/covid-zoe/processed_Mar10/processed_10032021.hdf5', 'r', 'source')
    #source = s.open_dataset('/mnt/data/jd21/data/March31.hdf5', 'r', 'source')
    output = s.open_dataset('/mnt/data/jd21/data_outtested2.hdf5', 'w', 'output')
    ds = DataStore()
    ts = str(datetime.now(timezone.utc))
    src_pat = source['patients']
    print(src_pat.keys())

    # Taking care of the result table
    #
    # src_pat = source['patients']
    # list_patid = ds.get_reader(src_pat['id'])
    # list_patcreate = ds.get_reader(src_pat['created_at'])
    # ind_pat = np.arange(len(list_patid), dtype=np.int64)
    # with utils.Timer('sorting_patients'):
    #     ind_pat = ds.dataset_sort(readers=(list_patid, list_patcreate), index=ind_pat)
    # out_pat = ds.get_or_create_group(output, 'patients')
    # # out_pat = output.create_group('patients')
    # new_patid = list_patid.get_writer(out_pat, 'id', ts)
    # new_patcreate = list_patcreate.get_writer(out_pat, 'created_at', ts, write_mode='overwrite')
    # with utils.Timer('applying sort'):
    #     ds.apply_sort(index=ind_pat, reader=list_patid, writer=new_patid)
    #     ds.apply_sort(index=ind_pat, reader=list_patcreate, writer=new_patcreate)
    # # Dropping duplicates of patient id rows
    # dup_filter = prst.filter_duplicate_fields(ds.get_reader(out_pat['id'])[:])
    # print(np.count_nonzero(dup_filter == True), len(list_patid))
    # for k in ('id', 'created_at'):
    #     new_reader = ds.get_reader(out_pat[k])
    #     flt_writer = new_reader.get_writer(out_pat, k, ts, write_mode='overwrite')
    #     x = ds.apply_filter(filter_to_apply=dup_filter, reader=new_reader, writer=flt_writer)
    #     print(len(x))
    #
    # # Same but for test
    src_test = source['tests']
    list_testid = src_test['patient_id']
    list_testcreate = src_test['created_at']

    src_test = source['tests']
    out_test = output.create_dataframe('test_count')

    print(src_test.keys())

    pat_id = src_test['patient_id']
    result = src_test['result']
    mechanism = src_test['mechanism']
    created_at = src_test['created_at']

    print('creating spans')
    spans = pat_id.get_spans()
    span_start = spans[0:-1]
    span_end = spans[1:]

    print('creating filter')
    clean_filt = np.zeros(len(pat_id), dtype='bool')
    print(len(pat_id))


    def clean_test(span_start, span_end, clean_filt,
                   fields=('patient_id', 'result', 'pcr_standard', 'date_effective_test')):
        dict_fields = {}
        print(clean_filt.sum())
        for f in fields:
            print(f)
            dict_fields[f] = src_test[f].data[:]
        dict_fields['date_eff'] = np.where(dict_fields['date_taken_specific'] > 0,
                                           dict_fields['date_taken_specific'],
                                           dict_fields['date_taken_between_start'])

        print('dict done')
        date_eff = dict_fields['date_eff']
        mechanism = dict_fields['mechanism']
        pat_tmp = dict_fields['patient_id']
        count_pat = 0
        print('starting loop')
        for i in range(len(span_start)):
            i_s = span_start[i]
            i_e = span_end[i]

            if i_e - i_s == 1:
                clean_filt[i_s] = 1
                count_pat += 1
                # print(np.sum(clean_filt), 'from unique')
            else:
                num_pat = len(np.unique(pat_tmp[i_s:i_e]))
                if num_pat > 1:
                    print('number of patid', num_pat)
                possible_dates = np.unique(date_eff[i_s:i_e])
                possible_mechanisms = np.unique(mechanism[i_s:i_e])
                current = 1
                for d in possible_dates:
                    for m in possible_mechanisms:
                        found = False
                        # print(d,m)
                        for j in reversed(range(i_s, i_e)):
                            # print(i_s, d, m, date_eff[j], mechanism[j])
                            if found == True:
                                clean_filt[j] = 0
                                continue
                            if date_eff[j] == d and mechanism[j] == m:
                                found = True
                                # print(d,m)
                                clean_filt[j] = 1
                                break
                        # print(np.sum(clean_filt))
                count_pat += 1
        print(count_pat)


    clean_test(span_start, span_end, clean_filt, fields=('patient_id', 'result', 'mechanism', 'date_taken_specific',
                                                         'date_taken_between_start', 'created_at'))

    # ind_test = np.arange(len(list_testid), dtype=np.int64)
    # with utils.Timer('sorting tests'):
    #     ind_test = ds.dataset_sort(readers=(list_testid, list_testcreate), index=ind_test)
    out_test = output.create_dataframe('tests', src_test)
    out_test.apply_filter(clean_filt)

            # reader = ds.get_reader(src_test[k])
            # writer = reader.get_writer(out_test, k, ts, write_mode='overwrite')
            # ds.apply_filter(clean_filt, reader, writer)
            # # ds.apply_sort(index=ind_test, reader=reader, writer=writer)

    # Filter for taken specific / date between (for the test table for now)
    # Create new field date_effective_test that is date taken specific and if not available take the average between date_end and date_start
    specdate_filter = out_test['date_taken_specific'].data[:] != 0
    date_spec = out_test['date_taken_specific'].data[:]
    date_start = out_test['date_taken_between_start'].data[:]
    date_end = out_test['date_taken_between_end'].data[:]
    date_fin = np.where(specdate_filter == True, date_spec, date_start + 0.5 * (date_end - date_start))
    det = s.create_timestamp(out_test, 'date_effective_test', ts)
    det.data.write(date_fin)
    #ds.get_timestamp_writer(out_test, 'date_effective_test', ts).write(date_fin)

    # Filtering only definite results

    results_raw = out_test['result'].data[:]
    results_filt = np.where(np.logical_or(results_raw == 4, results_raw == 3), True, False)
    # for k in out_test.keys():
    #     reader = ds.get_reader(out_test[k])
    #     writer = reader.get_writer(out_test, k, ts, write_mode='overwrite')
    #     ds.apply_filter(filter_to_apply=results_filt, reader=reader, writer=writer)
    out_test.apply_filter(results_filt)

    # Filter check
    sanity_filter = (date_fin == 0)
    print(np.sum(sanity_filter))

    # Joining patients and tests
    # test2pat = ds.get_index(target=ds.get_reader(out_pat['id']),foreign_key=ds.get_reader(out_test['patient_id']))
    # ds.join(destination_pkey=ds.get_reader(out_pat['id']), fkey_indices=test2pat)

    # Creating clean mechanism
    reader_mec = out_test['mechanism']
    reader_ftmec = out_test['mechanism_freetext']
    pcr_standard_answers = s.create_numeric(out_test, 'pcr_standard_answers', 'bool', ts)
    pcr_strong_inferred = s.create_numeric(out_test, 'pcr_strong_inferred', 'bool', ts)
    pcr_weak_inferred = s.create_numeric(out_test, 'pcr_weak_inferred', 'bool', ts)
    antibody_standard_answers = s.create_numeric(out_test, 'antibody_standard_answers', 'bool', ts)
    antibody_strong_inferred = s.create_numeric(out_test, 'antibody_strong_inferred', 'bool', ts)
    antibody_weak_inferred = s.create_numeric(out_test, 'antibody_weak_inferred', 'bool', ts)
    # pcr_standard_answers = ds.get_numeric_writer(out_test, 'pcr_standard_answers', 'bool', ts)
    # pcr_strong_inferred = ds.get_numeric_writer(out_test, 'pcr_strong_inferred', 'bool', ts)
    # pcr_weak_inferred = ds.get_numeric_writer(out_test, 'pcr_weak_inferred', 'bool', ts)
    # antibody_standard_answers = ds.get_numeric_writer(out_test, 'antibody_standard_answers', 'bool', ts)
    # antibody_strong_inferred = ds.get_numeric_writer(out_test, 'antibody_strong_inferred', 'bool', ts)
    # antibody_weak_inferred = ds.get_numeric_writer(out_test, 'antibody_weak_inferred', 'bool', ts)

    test_type_from_mechanism_v1(ds, reader_mec, reader_ftmec,
                                pcr_standard_answers, pcr_strong_inferred, pcr_weak_inferred,
                                antibody_standard_answers, antibody_strong_inferred, antibody_weak_inferred)

    # if 'mechanism_clean' not in out_test.keys():
    #     from exetera.covidspecific import data_schemas
    #
    #     data_schema = data_schemas.DataSchema(1)
    #     entry_clean = data_schema.test_categorical_maps['mechanism']
    #     print(entry_clean.strings_to_values)
    #     importer_mec = rw.LeakyCategoricalImporter(ds, out_test, 'mechanism_clean',
    #                                                  entry_clean.strings_to_values, entry_clean.out_of_range_label,ts)
    #     print(reader_mec[:10])
    #     importer_mec.write(reader_mec[:])

    # print(len(ds.get_reader(out_test['patient_id'])), len(ds.get_reader(out_test['mechanism_clean'])),
    #       len(date_fin))
    # Create a pandas DataFrame to join patient list and test table

    # mec_clean = ds.get_reader(out_test['mechanism_clean'])[:]

    reader_pcr_sa = out_test['pcr_standard_answers']
    reader_pcr_si = out_test['pcr_strong_inferred']
    reader_pcr_wi = out_test['pcr_weak_inferred']

    pcr_standard = reader_pcr_si.data[:] + reader_pcr_sa.data[:] + reader_pcr_wi.data[:]
    pcr_standard = np.where(pcr_standard > 0, np.ones_like(pcr_standard), np.zeros_like(pcr_standard))

    # writer = ds.get_numeric_writer(out_test, 'pcr_standard', dtype='bool', timestamp=ts, writemode='overwrite')
    # writer.write(pcr_standard)
    write = s.create_numeric(out_test, 'pcr_standard', 'bool', timestamp=ts)
    write.data.write(pcr_standard)

    # print(len(reader_mec), len(ds.get_reader(out_test['patient_id'])))
    # if 'mechanism_clean' not in out_test.keys():
    #     from exetera.covidspecific import data_schemas
    #
    #     data_schema = data_schemas.DataSchema(1)
    #     entry_clean = data_schema.test_categorical_maps['mechanism']
    #     print(entry_clean.strings_to_values)
    #     importer_mec = prst.LeakyCategoricalImporter(ds, out_test, 'mechanism_clean', ts,
    #                                                  entry_clean.strings_to_values, entry_clean.out_of_range_label)
    #     importer_mec.write(reader_mec[:])
    #
    # print(len(ds.get_reader(out_test['patient_id'])), len(ds.get_reader(out_test['mechanism_clean'])),
    #       len(date_fin))
    # Create a pandas DataFrame to join patient list and test table

    # mec_clean = ds.get_reader(out_test['mechanism_clean'])[:]
    # pcr_standard = np.where(np.logical_and(mec_clean > 0, mec_clean < 5), 1, 0)
    # writer = ds.get_numeric_writer(out_test, 'pcr_standard', dtype='bool', timestamp=ts, writemode='overwrite')
    # writer.write(pcr_standard)

    out_test_fin = output.create_dataframe('tests_fin')
    writers_dict = {}
    for k in ('patient_id', 'date_effective_test', 'result', 'pcr_standard', 'converted_test'):
        if k == 'converted_test':

            # values = np.zeros(len(ds.get_reader(out_test_fin['patient_id'])), dtype='bool')
            # writers_dict[k] = ds.get_numeric_writer(out_test_fin, k, timestamp=ts, dtype='bool',
            #                                         writemode='overwrite')
            values = np.zeros(len(out_test_fin['patient_id'].data), dtype='bool')
            writers_dict[k] = out_test_fin.create_numeric(k, 'bool', timestamp=ts).data

        else:
            # reader = ds.get_reader(out_test[k])
            # values = reader[:]
            # if k == 'result':
            #     values -= 3
            # writers_dict[k] = reader.get_writer(out_test_fin, k, ts, write_mode='overwrite')
            # print(len(values), k)
            values = out_test[k].data[:]
            if k == 'result':
                values -= 3
            writers_dict[k] = out_test[k].create_like(out_test_fin, k, ts).data
            print(len(values), k)
        writers_dict[k].write_part(values)

    # Taking care of the old test
    print('taking care of the old test')
    src_asmt = source['assessments']
    print(src_asmt.keys())

    # Remap had_covid_test to 0/1 2 to binary 0,1
    # tcp_flat = np.where(ds.get_reader(src_asmt['tested_covid_positive'])[:] < 1, 0, 1)
    # spans = ds.get_spans(ds.get_reader(src_asmt['patient_id']))
    tcp_flat = np.where(src_asmt['tested_covid_positive'].data[:] < 1, 0, 1)
    spans = src_asmt['patient_id'].get_spans()
    # Get the first index at which the hct field is maximum
    firstnz_tcp_ind = ds.apply_spans_index_of_max(spans, tcp_flat)
    # Get the index of first element of patient_id when sorted
    first_hct_ind = spans[:-1]
    filt_tl = first_hct_ind != firstnz_tcp_ind
    # Get the indices for which hct changed value (indicating that test happened after the first input)
    sel_max_ind = ds.apply_filter(filter_to_apply=filt_tl, reader=firstnz_tcp_ind)

    # Get the index at which test is maximum and for which that hct is possible
    max_tcp_ind = ds.apply_spans_index_of_max(spans, src_asmt['tested_covid_positive'].data[:])
    # filt_max_test = ds.apply_indices(filt_tl, max_tcp )
    sel_max_tcp = ds.apply_indices(filt_tl, max_tcp_ind)
    sel_maxtcp_ind = ds.apply_filter(filter_to_apply=filt_tl, reader=max_tcp_ind)

    # Define usable assessments with correct test based on previous filter on indices
    # if 'usable_asmt_tests' not in output.keys():
    #     usable_asmt_tests = output.create_dataframe('usable_asmt_tests')
    # else:
    #     usable_asmt_tests = output['usable_asmt_tests']
    # for k in ('id', 'patient_id', 'created_at', 'had_covid_test'):
    #     reader = ds.get_reader(src_asmt[k])
    #     writer = reader.get_writer(usable_asmt_tests, k, ts, write_mode='overwrite')
    #     ds.apply_indices(sel_max_ind, reader=reader, writer=writer)
    #     # print(ds.get_reader(usable_asmt_tests[k])[0])
    usable_asmt_tests = output.create_dataframe('usable_asmt_tests')
    for k in ('id', 'patient_id', 'created_at', 'had_covid_test'):
        fld = src_asmt[k].create_like(usable_asmt_tests, k, ts)
        src_asmt[k].apply_index(sel_max_ind, target=fld)


    # reader = ds.get_reader(src_asmt['created_at'])
    # writer = reader.get_writer(usable_asmt_tests, 'eff_result_time', ts, write_mode='overwrite')
    # ds.apply_indices(sel_maxtcp_ind, reader, writer)
    fld = src_asmt['created_at'].create_like(usable_asmt_tests, 'eff_result_time', ts)
    src_asmt['created_at'].apply_index(sel_maxtcp_ind, target=fld)


    # reader = ds.get_reader(src_asmt['tested_covid_positive'])
    # writer = reader.get_writer(usable_asmt_tests, 'eff_result', ts, write_mode='overwrite')
    # ds.apply_indices(sel_maxtcp_ind, reader, writer)
    fld = src_asmt['tested_covid_positive'].create_like(usable_asmt_tests, 'eff_result', ts)
    src_asmt['tested_covid_positive'].apply_index(sel_maxtcp_ind, target = fld)


    for k in ('tested_covid_positive',):
        # reader = ds.get_reader(src_asmt[k])
        # writer = reader.get_writer(usable_asmt_tests, k, ts, write_mode='overwrite')
        # ds.apply_indices(sel_max_tcp, reader, writer)
        fld = src_asmt[k].create_like(usable_asmt_tests, k, ts)
        src_asmt[k].apply_index(sel_max_tcp, target = fld)
        # print(ds.get_reader(usable_asmt_tests[k])[0])

    # Making sure that the test is definite (either positive or negative)
    print('filter definite test answers')
    filt_deftest = usable_asmt_tests['tested_covid_positive'].data[:] > 1
    print(np.sum(filt_deftest), len(usable_asmt_tests['tested_covid_positive'].data))
    # print(len(ds.get_reader(usable_asmt_tests['patient_id'])))
    for k in (
            'id', 'patient_id', 'created_at', 'had_covid_test', 'tested_covid_positive', 'eff_result_time',
            'eff_result'):
        # reader = ds.get_reader(usable_asmt_tests[k])
        # writer = reader.get_writer(usable_asmt_tests, k, ts, write_mode='overwrite')
        # ds.apply_filter(filt_deftest, reader, writer)
        fld = usable_asmt_tests[k]
        usable_asmt_tests[k].apply_filter(filt_deftest, target=fld)

    # Getting difference between created at (max of hct date) and max of test result (eff_result_time)

    reader_hct = usable_asmt_tests['created_at'].data[:]
    reader_tcp = usable_asmt_tests['eff_result_time'].data[:]
    with utils.Timer('doing delta time'):
        delta_time = reader_tcp - reader_hct
        delta_days = delta_time / 86400
    print(delta_days[:10], delta_time[:10])
    #writer = ds.get_numeric_writer(usable_asmt_tests, 'delta_days_test', 'float32', ts, writemode='overwrite')
    writer = usable_asmt_tests.create_numeric('delta_days_test', 'float32', ts).data
    writer.write(delta_days)
    print(delta_days.shape, ' size of delta days')

    # Final day of test

    date_final_test = np.where(delta_days < 7, reader_hct, reader_tcp - 2 * 86400)
    # writer = ds.get_timestamp_writer(usable_asmt_tests, 'date_final_test', ts, writemode='overwrite')
    writer = usable_asmt_tests.create_timestamp('date_final_test', ts).data
    writer.write(date_final_test)
    # print(ds.get_reader(usable_asmt_tests['date_final_test'])[:10], date_final_test[:10])

    pcr_standard = np.ones(len(usable_asmt_tests['patient_id'].data))
    # writer = ds.get_numeric_writer(usable_asmt_tests, 'pcr_standard', dtype='bool', timestamp=ts,
    #                               writemode='overwrite')
    writer = usable_asmt_tests.create_numeric('pcr_standard', 'bool', timestamp=ts).data
    writer.write(np.array(pcr_standard, dtype=bool))

    list_init = ('patient_id', 'date_final_test', 'tested_covid_positive', 'pcr_standard')
    list_final = ('patient_id', 'date_effective_test', 'result', 'pcr_standard')
    # Join
    for (i, f) in zip(list_init, list_final):
        reader = usable_asmt_tests[i].data
        values = reader[:]
        if f == 'result':
            values -= 2
        # writers_dict[f] = reader.get_writer(out_test_fin, f, ts)
        print(len(values), f)
        if len(values) > 0:
            writers_dict[f].write(values)
    if len(usable_asmt_tests['patient_id'].data) > 0:
        writers_dict['converted_test'].write(np.ones(len(usable_asmt_tests['patient_id'].data), dtype='bool'))
    converted_fin = out_test_fin['converted_test'].data
    result_fin = out_test_fin['result'].data[:]
    pat_id_fin = out_test_fin['patient_id'].data[:]

    out_asmt_init = output.create_dataframe('init_asmt')
    filt_asmt = prst.foreign_key_is_in_primary_key(out_test_fin['patient_id'].data[:],
                                                   src_asmt['patient_id'].data[:])
    print(np.sum(filt_asmt), 'kept with filt_asmt')
    for k in list_symptoms + ['created_at', 'patient_id']:
        # reader = ds.get_reader(src_asmt[k])
        # writer = reader.get_writer(out_asmt_init, k, ts)
        # ds.apply_filter(filt_asmt, reader, writer)
        fld = src_asmt[k].create_like(out_asmt_init, k, timestamp=ts)
        src_asmt[k].apply_filter(filt_asmt, target=fld)

    filt_asmt2 = prst.foreign_key_is_in_primary_key(out_asmt_init['patient_id'].data[:],
                                                    out_test_fin['patient_id'].data[:])

    for k in out_test_fin.keys():
        # reader = ds.get_reader(out_test_fin[k])
        # writer = reader.get_writer(out_test_fin, k, ts, write_mode='overwrite')
        # ds.apply_filter(filt_asmt2, reader, writer)
        out_test_fin[k].apply_filter(filt_asmt2, in_place=True)

    list_patid = out_test_fin['patient_id'].data
    list_patcreate = out_test_fin['date_effective_test'].data
    ind_pat = np.arange(len(list_patid), dtype=np.int64)
    with utils.Timer('sorting_patients'):
        ind_pat = ds.dataset_sort(readers=(list_patid, list_patcreate), index=ind_pat)
    out_test_fin2 = output.create_dataframe('test_fin2')
    with utils.Timer('applying sort'):
        for k in out_test_fin.keys():
            #reader = ds.get_reader(out_test_fin[k])
            # writer = reader.get_writer(out_test_fin2, k, ts)
            # ds.apply_sort(ind_pat, reader, writer)
            fld = out_test_fin[k].create_like(out_test_fin2, k, ts)
            out_test_fin[k].apply_index(ind_pat, target=fld)


    pat_init = out_asmt_init['patient_id'].data
    filt_date = np.zeros(len(pat_init), dtype='bool')
    span_testpat = np.zeros(len(pat_init))
    filt_testwithasmt = np.zeros(len(out_test_fin2['patient_id'].data))

    print(len(np.unique(out_asmt_init['patient_id'].data[:])),
          len(np.unique(out_test_fin2['patient_id'].data[:])))

    spans_test = out_test_fin2['patient_id'].get_spans()
    spans_asmt = out_asmt_init['patient_id'].get_spans()

    print('Create filtering of dates')

    dict_max = {}
    for f in list_symptoms:
        dict_max[f] = []
    dict_max['healthy_first'] = []

    sum_symp = np.zeros(len(out_asmt_init['patient_id'].data))
    for k in list_symptoms:
        values = out_asmt_init[k].data[:]
        print(k, np.unique(values))
        values = np.where(values > 1, np.ones_like(values), np.zeros_like(values))
        sum_symp += values

    first_sum = sum_symp[spans_asmt[:-1]]
    first_healthy = first_sum == 0
    print(np.sum(first_healthy), ' number of start healthy')
    filt_healthy_first = np.zeros(len(out_asmt_init['patient_id'].data))


    def date_filtering(spans_test, spans_asmt, filt_date, span_testpat, filt_testwithasmt, dict_max, filt_hf):
        count_ta = 0
        spans_start_test = spans_test[:-1]
        spans_end_test = spans_test[1:]
        spans_start_asmt = spans_asmt[:-1]
        spans_end_asmt = spans_asmt[1:]

        date_eff = out_test_fin['date_effective_test'].data[:]
        date_update = out_asmt_init['created_at'].data[:]

        dict_symp = {}
        for f in list_symptoms:
            dict_symp[f] = out_asmt_init[f].data[:]

        for i in range(len(spans_start_test)):
            i_sa = spans_start_asmt[i]
            i_ea = spans_end_asmt[i]
            i_st = spans_start_test[i]
            i_et = spans_end_test[i]
            filt_hf[i_sa:i_ea] = first_healthy[i]
            dates_test = date_eff[i_st:i_et]
            dates_asmt = date_update[i_sa:i_ea]
            for (e, d) in enumerate(dates_test):
                dates_pos = np.where(np.logical_and(dates_asmt <= d, dates_asmt >= d - 86400 * 14), 1, 0)
                filt_date[i_sa:i_ea] += dates_pos.astype('bool')

                # span_testpat[i_sa:i_ea] = 500000 * (e+1) + i
                if np.sum(dates_pos) > 0:  # and first_healthy[i]==1:
                    filt_testwithasmt[e + i_st] = 1
                    dict_max['healthy_first'].append(first_healthy[i])
                    for f in list_symptoms:
                        values = dict_symp[f][i_sa:i_ea]
                        values_pos = values[dates_pos == 1]
                        dict_max[f].append(np.max(values_pos))
                    count_ta += 1
        print(count_ta)


    with utils.Timer('pruning to 14 days before test'):
        date_filtering(spans_test, spans_asmt, filt_date, span_testpat, filt_testwithasmt, dict_max,
                       filt_healthy_first)
    out_asmt_filt = output.create_dataframe('asmt_filtered')
    for f in list_symptoms + ['created_at', ]:
        # reader = ds.get_reader(out_asmt_init[f])
        # writer = reader.get_writer(out_asmt_filt, f, ts)
        # # ds.apply_filter((filt_date*filt_healthy_first).astype(bool),reader,writer)
        # ds.apply_filter(filt_date, reader, writer)
        fld = out_asmt_init[f].create_like(out_asmt_filt, f, ts)
        out_asmt_init[f].apply_filter(filt_date, target=fld)

    print('pruning done')
    # writer = ds.get_numeric_writer(out_asmt_filt,'healthy_first',dtype='bool',timestamp=ts,writemode='overwrite')
    # ds.apply_filter(filt_date,filt_healthy_first.astype(bool),writer)

    out_test_filt = output.create_dataframe('test_filtered')

    for k in out_test_fin2.keys():
        # reader = ds.get_reader(out_test_fin2[k])
        # writer = reader.get_writer(out_test_filt, k, ts)
        # ds.apply_filter(filt_testwithasmt.astype('bool'), reader, writer)
        fld = out_test_fin2[k].create_like(out_test_filt, k, ts)
        out_test_fin2[k].apply_filter(filt_testwithasmt.astype('bool'), target=fld)

    for k in out_test_filt.keys():
        dict_max[k] = out_test_filt[k].data[:]

    # dict_max['healthy_first'] = ds.get_reader(out_asmt_filt['healthy_first'])[:]
    df_maxbeftest = pd.DataFrame.from_dict(dict_max)
    df_maxbeftest.to_csv('/mnt/data/jd21/MaxBefTestForLongAll.csv')

    for k in list_symptoms:
        #writer = ds.get_numeric_writer(out_test_filt, k, timestamp=ts, dtype='int64')
        if k in out_test_filt.keys():
            writer = out_test_filt[k].data
        else:
            writer = out_test_filt.create_numeric(k, 'int64', timestamp=ts).data
        writer.write(np.asarray(dict_max[k]))

    # out_test_pcr = output.create_group('only_pcr')
    src_pat = source['patients']
    out_pat = output.create_dataframe('patients_characteristics')
    pat_id = src_pat['id'].data
    filt_patfin = prst.foreign_key_is_in_primary_key(out_test_filt['patient_id'].data[:], pat_id[:])
    list_fields = ('id', 'year_of_birth', 'gender', 'healthcare_professional')
    dict_pat = {}
    for k in list_fields:
        # reader = ds.get_reader(src_pat[k])
        # writer = reader.get_writer(out_pat, k, ts)
        # ds.apply_filter(filt_patfin, reader, writer)
        fld = src_pat[k].create_like(out_pat, k, ts)
        src_pat[k].apply_filter(filt_patfin, target=fld)
        dict_pat[k] = out_pat[k].data[:]

    df_pat = pd.DataFrame.from_dict(dict_pat)
    df_pat.to_csv('/mnt/data/jd21/PatBefTestForLongAll.csv')




