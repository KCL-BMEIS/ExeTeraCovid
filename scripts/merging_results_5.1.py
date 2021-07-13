import itertools
import numpy as np
import pandas as pd
import exetera as hy
import h5py
import glob
from datetime import datetime, timezone


from exetera.core.session import Session
from exetera.core.persistence import DataStore
from exetera.core import utils, dataframe, dataset
from exetera.core import persistence as prst
from exetera.core import readerwriter as rw
from exeteracovid.algorithms.test_type_from_mechanism import test_type_from_mechanism_v1


ds = DataStore()
ts = str(datetime.now(timezone.utc))

path = '/home/jd21/data'
list_symptoms =['abdominal_pain', 'altered_smell', 'blisters_on_feet', 'brain_fog',
       'chest_pain', 'chills_or_shivers','delirium', 'diarrhoea',
       'diarrhoea_frequency', 'dizzy_light_headed', 'ear_ringing', 'earache',
       'eye_soreness', 'fatigue', 'feeling_down', 'fever', 'hair_loss',
       'headache', 'headache_frequency','hoarse_voice',
       'irregular_heartbeat', 'loss_of_smell', 'nausea','persistent_cough', 'rash', 'red_welts_on_face_or_lips', 'runny_nose',
       'shortness_of_breath', 'skin_burning', 'skipped_meals', 'sneezing',
       'sore_throat', 'swollen_glands', 'typical_hayfever', 'unusual_muscle_pains']

with Session() as s:

    source = s.open_dataset('/home/jd21/data/processed_May17_processed.hdf5', 'r', 'src')
    output = s.open_dataset('/home/jd21/data/May17_processed_mrslt.hdf5', 'w', 'out')
    ds = DataStore()
    ts = str(datetime.now(timezone.utc))

    # # Same but for test
    src_test = source['tests']
    list_testid = src_test['patient_id']
    list_testcreate = src_test['created_at']
    out_test = output.create_dataframe('tests')
    with utils.Timer('applying sort'):
        for k in src_test.keys():
            dataframe.copy(src_test[k], out_test, k)
            # reader = ds.get_reader(src_test[k])
            # writer = reader.get_writer(out_test, k, ts, write_mode='overwrite')
            # writer.write(reader[:])

    # Filter for taken specific / date between (for the test table for now)
    # Create new field date_effective_test that is date taken specific and if not available take the average between date_end and date_start
    specdate_filter = out_test['date_taken_specific'].data[:] != 0
    date_spec =  out_test['date_taken_specific'].data[:]
    date_start =  out_test['date_taken_between_start'].data[:]
    date_end =  out_test['date_taken_between_end'].data[:]
    date_fin = np.where(specdate_filter == True, date_spec, date_start + 0.5 * (date_end - date_start))
    #ds.get_timestamp_writer(out_test, 'date_effective_test', ts).write(date_fin)
    date_effective_test = out_test.create_timestamp('date_effective_test')
    date_effective_test.data.write(date_fin)

    # Filtering only definite results

    results_raw = out_test['result'].data[:]
    results_filt = np.where(np.logical_or(results_raw == 4, results_raw == 3), True, False)
    for k in out_test.keys():
        out_test[k].apply_filter(results_filt, in_place=True)
        # reader = ds.get_reader(out_test[k])
        # writer = reader.get_writer(out_test, k, ts, write_mode='overwrite')
        # ds.apply_filter(filter_to_apply=results_filt, reader=reader, writer=writer)

    # Filter check
    sanity_filter = (date_fin == 0)
    print(np.sum(sanity_filter))

    # Creating clean mechanism
    reader_mec = out_test['mechanism'].data
    s_reader_mec = s.get(out_test['mechanism'])

    print(len(reader_mec), len(out_test['patient_id'].data))

    reader_ftmec = out_test['mechanism_freetext'].data
    s_reader_ftmec = s.get(out_test['mechanism_freetext'])

    # pcr_standard_answers = ds.get_numeric_writer(out_test, 'pcr_standard_answers', 'bool', ts)
    # pcr_strong_inferred = ds.get_numeric_writer(out_test, 'pcr_strong_inferred', 'bool', ts)
    # pcr_weak_inferred = ds.get_numeric_writer(out_test, 'pcr_weak_inferred', 'bool', ts)
    # antibody_standard_answers = ds.get_numeric_writer(out_test, 'antibody_standard_answers', 'bool', ts)
    # antibody_strong_inferred = ds.get_numeric_writer(out_test, 'antibody_strong_inferred', 'bool', ts)
    # antibody_weak_inferred = ds.get_numeric_writer(out_test, 'antibody_weak_inferred', 'bool', ts)
    pcr_standard_answers = out_test.create_numeric('pcr_standard_answers', 'bool')
    pcr_strong_inferred = out_test.create_numeric('pcr_strong_inferred', 'bool')
    pcr_weak_inferred = out_test.create_numeric('pcr_weak_inferred', 'bool')
    antibody_standard_answers = out_test.create_numeric('antibody_standard_answers', 'bool')
    antibody_strong_inferred = out_test.create_numeric('antibody_strong_inferred', 'bool')
    antibody_weak_inferred = out_test.create_numeric('antibody_weak_inferred', 'bool')

    t_pids = s.get(out_test['patient_id'])
    with utils.Timer('getting test mechanism filter for pcr and antibody', new_line=True):
        pcr_standard_answers = np.zeros(len(t_pids), dtype=np.bool)
        pcr_strong_inferred = np.zeros(len(t_pids), dtype=np.bool)
        pcr_weak_inferred = np.zeros(len(t_pids), dtype=np.bool)
        antibody_standard_answers = np.zeros(len(t_pids), dtype=np.bool)
        antibody_strong_inferred = np.zeros(len(t_pids), dtype=np.bool)
        antibody_weak_inferred = np.zeros(len(t_pids), dtype=np.bool)

    test_type_from_mechanism_v1(ds, s_reader_mec, s_reader_ftmec,
                                pcr_standard_answers, pcr_strong_inferred, pcr_weak_inferred,
                                antibody_standard_answers, antibody_strong_inferred, antibody_weak_inferred)

    reader_pcr_sa = s.get(out_test['pcr_standard_answers'])
    reader_pcr_si = s.get(out_test['pcr_strong_inferred'])
    reader_pcr_wi = s.get(out_test['pcr_weak_inferred'])

    pcr_standard = pcr_strong_inferred + pcr_standard_answers + pcr_weak_inferred
    pcr_standard = np.where(pcr_standard > 0, np.ones_like(pcr_standard), np.zeros_like(pcr_standard))

    #writer = ds.get_numeric_writer(out_test, 'pcr_standard', dtype='bool', timestamp=ts, writemode='overwrite')
    writer = out_test.create_numeric('pcr_standard', 'bool')
    writer.data.write(pcr_standard)

    out_test_fin = output.create_dataframe('tests_fin')
    writers_dict = {}
    for k in ('patient_id', 'date_effective_test', 'result', 'pcr_standard', 'converted_test'):
        if k == 'converted_test':
            values = np.zeros(len(out_test_fin['patient_id'].data), dtype='bool')
            #writers_dict[k] = ds.get_numeric_writer(out_test_fin, k, timestamp=ts, dtype='bool',
                                                    #writemode='overwrite')
            writers_dict[k] = out_test_fin.create_numeric(k, 'bool')
        else:
            values = out_test[k].data[:]
            if k == 'result':
                values -= 3
            #writers_dict[k] = reader.get_writer(out_test_fin, k, ts, write_mode='overwrite')
            writers_dict[k] = out_test[k].create_like(out_test_fin, k)
            print(len(values), k)
        writers_dict[k].data.write(values)

    # Taking care of the old test
    src_asmt = source['assessments']
    print(src_asmt.keys())

    # Remap had_covid_test to 0/1 2 to binary 0,1
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
    if 'usable_asmt_tests' not in output.keys():
        usable_asmt_tests = output.create_group('usable_asmt_tests')
    else:
        usable_asmt_tests = output['usable_asmt_tests']
    for k in ('id', 'patient_id', 'created_at', 'had_covid_test'):
        src_asmt[k].create_like(usable_asmt_tests, k)
        src_asmt[k].apply_index(sel_max_ind, target=usable_asmt_tests[k])
        print(usable_asmt_tests[k].data[0])
        # reader = ds.get_reader(src_asmt[k])
        # writer = reader.get_writer(usable_asmt_tests, k, ts, write_mode='overwrite')
        # ds.apply_indices(sel_max_ind, reader=reader, writer=writer)
        # print(ds.get_reader(usable_asmt_tests[k])[0])

    src_asmt['created_at'].create_like(usable_asmt_tests, 'eff_result_time')
    src_asmt['created_at'].apply_index(sel_maxtcp_ind, target=usable_asmt_tests['eff_result_time'])
    # reader = ds.get_reader(src_asmt['created_at'])
    # writer = reader.get_writer(usable_asmt_tests, 'eff_result_time', ts, write_mode='overwrite')
    # ds.apply_indices(sel_maxtcp_ind, reader, writer)

    src_asmt['tested_covid_positive'].create_like(usable_asmt_tests, 'eff_result')
    src_asmt['tested_covid_positive'].apply_index(sel_maxtcp_ind, target=usable_asmt_tests['eff_result'])
    # reader = ds.get_reader(src_asmt['tested_covid_positive'])
    # writer = reader.get_writer(usable_asmt_tests, 'eff_result', ts, write_mode='overwrite')
    # ds.apply_indices(sel_maxtcp_ind, reader, writer)

    for k in ('tested_covid_positive',):
        src_asmt[k].create_like(usable_asmt_tests, k)
        src_asmt[k].apply_index(sel_max_tcp, target=usable_asmt_tests[k])
        # reader = ds.get_reader(src_asmt[k])
        # writer = reader.get_writer(usable_asmt_tests, k, ts, write_mode='overwrite')
        # ds.apply_indices(sel_max_tcp, reader, writer)

    # Making sure that the test is definite (either positive or negative)
    filt_deftest = usable_asmt_tests['tested_covid_positive'].data[:] > 1
    # print(len(ds.get_reader(usable_asmt_tests['patient_id'])))
    for k in (
            'id', 'patient_id', 'created_at', 'had_covid_test', 'tested_covid_positive', 'eff_result_time',
            'eff_result'):
        usable_asmt_tests[k].apply_filter(filt_deftest, in_place=True)
        # reader = ds.get_reader(usable_asmt_tests[k])
        # writer = reader.get_writer(usable_asmt_tests, k, ts, write_mode='overwrite')
        # ds.apply_filter(filt_deftest, reader, writer)

    # Getting difference between created at (max of hct date) and max of test result (eff_result_time)
    reader_hct = usable_asmt_tests['created_at'].data[:]
    reader_tcp = usable_asmt_tests['eff_result_time'].data[:]
    with utils.Timer('doing delta time'):
        delta_time = reader_tcp - reader_hct
        delta_days = delta_time / 86400
    print(delta_days[:10], delta_time[:10])
    writer = usable_asmt_tests.create_numeric('delta_days_test', 'float32')
    writer.data.write(delta_days)

    # Final day of test
    date_final_test = np.where(delta_days < 7, reader_hct, reader_tcp - 2 * 86400)
    writer = usable_asmt_tests.create_timestamp('date_final_test')
    writer.data.write(date_final_test)
    # print(ds.get_reader(usable_asmt_tests['date_final_test'])[:10], date_final_test[:10])

    pcr_standard = np.ones(len(usable_asmt_tests['patient_id'].data))
    writer = usable_asmt_tests.create_numeric('pcr_standard', 'int')
    writer.data.write(pcr_standard)

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
        writers_dict[f].data.write(values)
    writers_dict['converted_test'].data.write(np.ones(len(usable_asmt_tests['patient_id'].data), dtype='bool'))
    converted_fin = out_test_fin['converted_test'].data
    result_fin = out_test_fin['result'].data[:]
    pat_id_fin = out_test_fin['patient_id'].data[:]
    filt_pos = result_fin >= 0

    out_pos = output.create_dataframe('out_pos')
    for k in out_test_fin.keys():
        out_test_fin[k].create_like(out_pos, k)
        out_test_fin[k].apply_filter(filt_pos, target=out_pos[k])
        print(k, len(out_test_fin[k].data), len(filt_pos))
        # reader = ds.get_reader(out_test_fin[k])
        # writer = reader.get_writer(out_pos, k, ts)
        # ds.apply_filter(filt_pos, reader, writer)
    #pat_pos = ds.get_reader(out_pos['patient_id'])

    dataset.copy(out_pos, output, 'out_pos_copy')
    # dict_test = {}
    # for k in out_pos.keys():
    #     dict_test[k] = out_pos[k].data[:]
    # df_test = pd.DataFrame.from_dict(dict_test)
    # df_test.to_csv(path + '/TestedPositiveTestDetails.csv')
    # del dict_test
    # del df_test

    #pat_id_all = src_asmt['patient_id'].data[:]

    with utils.Timer('Mapping index asmt to pos only'):
        test2pat = prst.foreign_key_is_in_primary_key(out_pos['patient_id'].data[:],
                                                      foreign_key=src_asmt['patient_id'].data[:])


    for f in ['created_at', 'patient_id', 'treatment', 'other_symptoms', 'country_code', 'location',
              'updated_at'] + list_symptoms:
        print(f)
        if(f in list(out_pos.keys())):
            out_pos[f].data.clear()
            src_asmt[f].apply_filter(test2pat, target=out_pos[f])
        else:
            src_asmt[f].create_like(out_pos, f)
            src_asmt[f].apply_filter(test2pat, target=out_pos[f])
        # reader = ds.get_reader(src_asmt[f])
        # writer = reader.get_writer(out_pos,f,ts, write_mode='overwrite')
        # ds.apply_filter(test2pat, reader, writer)

    # print(len(np.unique(ds.get_reader(out_pos['patient_id'])[:])), len(np.unique(pat_pos[:])))
    print("skip unique")

    #  this is duplicated with 265-273
    # for k in list_symptoms:
    #     print(k)
    #     if k in list(out_pos.keys()):
    #         src_asmt[k].apply_filter(test2pat, target=out_pos[k])
    #     else:
    #         src_asmt[k].create_like(out_pos, k)
    #         src_asmt[k].apply_filter(test2pat, target=out_pos[k])
        # reader = ds.get_reader(src_asmt[k])
        # writer = reader.get_writer(out_pos, k,ts,write_mode='overwrite')
        # ds.apply_filter(test2pat, reader,writer)

    sum_symp = np.zeros(len(out_pos['patient_id'].data))
    for k in list_symptoms:
        values = out_pos[k].data[:]
        if k == 'fatigue' or k == 'shortness_of_breath':
            values = np.where(values > 2, np.ones_like(values), np.zeros_like(values))
        else:
            values = np.where(values > 1, np.ones_like(values), np.zeros_like(values))
        sum_symp += values

    out_pos.create_numeric('sum_symp', 'int').data.write(sum_symp)
    # writer = ds.get_numeric_writer(out_pos, 'sum_symp', dtype='int', timestamp=ts, writemode='overwrite')
    # writer.write(sum_symp)

    symp_flat = np.where(out_pos['sum_symp'].data[:] < 1, 0, 1)
    spans = out_pos['patient_id'].get_spans()
    # Get the first index at which the hct field is maximum
    firstnz_symp_ind = ds.apply_spans_index_of_max(spans, symp_flat)
    max_symp_check = symp_flat[firstnz_symp_ind]
    # Get the index of first element of patient_id when sorted
    first_symp_ind = spans[:-1]
    filt_asymptomatic = max_symp_check == 0
    filt_firsthh_symp = first_symp_ind != firstnz_symp_ind
    print('Number asymptomatic is ', len(spans) - 1 - np.sum(max_symp_check), np.sum(filt_asymptomatic))
    print('Number not healthy first is ', len(spans) - 1 - np.sum(filt_firsthh_symp))
    print('Number definitie positive is', len(spans) - 1)
    spans_valid = ds.apply_filter(filt_firsthh_symp, first_symp_ind)
    pat_sel = ds.apply_indices(spans_valid, out_pos['patient_id'].data[:])
    filt_sel = prst.foreign_key_is_in_primary_key(pat_sel, out_pos['patient_id'].data[:])

    spans_asymp = ds.apply_filter(filt_asymptomatic, first_symp_ind)
    pat_asymp = out_pos['patient_id'].apply_index(spans_asymp)
    #pat_asymp = ds.apply_indices(spans_asymp, ds.get_reader(out_pos['patient_id']))
    filt_pata = prst.foreign_key_is_in_primary_key(pat_asymp.data[:], out_pos['patient_id'].data[:])

    out_pos_hs = output.create_dataframe('out_pos_hs')
    for k in list_symptoms + ['created_at', 'patient_id', 'sum_symp', 'country_code', 'location', 'treatment',
                              'updated_at']:
        print(k)
        out_pos[k].create_like(out_pos_hs, k)
        out_pos[k].apply_filter(filt_sel, target=out_pos_hs[k])
        # reader = ds.get_reader(out_pos[k])
        # writer = reader.get_writer(out_pos_hs, k, ts)
        # ds.apply_filter(filt_sel, reader, writer)

    # dict_final = {}
    # for k in out_pos_hs.keys():
    #     dict_final[k] = out_pos_hs[k].data[:]
    #
    # df_final = pd.DataFrame.from_dict(dict_final)
    # df_final.to_csv(path + '/PositiveSympStartHealthyAllSymptoms.csv')
    # del dict_final
    # del df_final

    print('out_pos_asymp')
    out_pos_as = output.create_dataframe('out_pos_asymp')
    for k in list_symptoms + ['created_at', 'patient_id', 'sum_symp', 'country_code', 'location',
                              'treatment']:
        out_pos[k].create_like(out_pos_as, k)
        out_pos[k].apply_filter(filt_pata, target=out_pos_as[k])
        # reader = ds.get_reader(out_pos[k])
        # writer = reader.get_writer(out_pos_as, k, ts)
        # ds.apply_filter(filt_pata, reader, writer)

    # dict_finala = {}
    # for k in out_pos_as.keys():
    #     dict_finala[k] = out_pos_as[k].data[:]
    #
    # df_finala = pd.DataFrame.from_dict(dict_finala)
    # df_finala.to_csv(path + '/PositiveAsympAllSymptoms.csv')
    # del dict_finala
    # del df_finala

    # Based on the final selected patient_id, select the appropriate rows of the patient_table
    src_pat = source['patients']
    filt_pat = prst.foreign_key_is_in_primary_key(out_pos_hs['patient_id'].data[:], src_pat['id'].data[:])
    list_interest = ['has_cancer', 'has_diabetes', 'has_lung_disease', 'has_heart_disease', 'has_kidney_disease',
                     'has_asthma',
                     'race_is_other', 'race_is_prefer_not_to_say', 'race_is_uk_asian', 'race_is_uk_black',
                     'race_is_uk_chinese', 'race_is_uk_middle_eastern', 'race_is_uk_mixed_other',
                     'race_is_uk_mixed_white_black', 'race_is_uk_white', 'race_is_us_asian', 'race_is_us_black',
                     'race_is_us_hawaiian_pacific', 'race_is_us_indian_native', 'race_is_us_white', 'race_other',
                     'year_of_birth', 'is_smoker', 'smoker_status', 'bmi_clean', 'is_in_uk_twins',
                     'healthcare_professional', 'gender', 'id', 'blood_group', 'lsoa11cd', 'already_had_covid']
    out_pat = output.create_dataframe('patient_pos')
    print('patient_pos')
    for k in list_interest:
        src_pat[k].create_like(out_pat, k)
        src_pat[k].apply_filter(filt_pat, target=out_pat[k])
        # reader = ds.get_reader(src_pat[k])
        # writer = reader.get_writer(out_pat, k, ts)
        # ds.apply_filter(filt_pat, reader, writer)

    # dict_pat = {}
    # for k in list_interest:
    #     values = out_pat[k].data[:]
    #     dict_pat[k] = values
    #
    # df_pat = pd.DataFrame.from_dict(dict_pat)
    # df_pat.to_csv(path + '/PositiveSympStartHealthy_PatDetails.csv')
    # del dict_pat
    # del df_pat

    spans_asymp = ds.apply_filter(filt_asymptomatic, first_symp_ind)
    #pat_asymp = ds.apply_indices(spans_asymp, ds.get_reader(out_pos['patient_id']))
    pat_asymp = out_pos['patient_id'].apply_index(spans_asymp)
    filt_asymp = prst.foreign_key_is_in_primary_key(pat_asymp.data[:], src_pat['id'].data[:])
    out_pat_asymp = output.create_dataframe('patient_asymp')
    for k in list_interest:
        src_pat[k].create_like(out_pat_asymp, k)
        src_pat[k].apply_filter(filt_asymp, target=out_pat_asymp[k])
        # reader = ds.get_reader(src_pat[k])
        # writer = reader.get_writer(out_pat_asymp, k, ts)
        # ds.apply_filter(filt_asymp, reader, writer)

    # dict_pata = {}
    # for k in list_interest:
    #     values = out_pat_asymp[k].data[:]
    #     dict_pata[k] = values
    #
    # df_pata = pd.DataFrame.from_dict(dict_pata)
    # df_pata.to_csv(path + '/PositiveAsymp_PatDetails.csv')
