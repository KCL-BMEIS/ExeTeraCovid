from datetime import datetime
import numpy as np

import exetera.core.session as sess
from exetera.core import dataframe


ADATA = '/home/jd21/data/processed_May17_processed.hdf5'
VDATA = '/home/jd21/data/vacc.0603.h5'
DSTDATA = '/home/jd21/data/full_merge.h5'


def asmt_merge_vacc():
    """
    Merge assessment df with vaccine dataframe, filter out subject has a healthy assessments before vaccine date
    """
    with sess.Session() as s:
        # open related datasets
        src = s.open_dataset(ADATA, 'r', 'asmt')
        asmt = src['assessments']
        vacc = s.open_dataset(VDATA, 'r', 'vacc')
        dst = s.open_dataset(DSTDATA, 'w', 'dst')

        #filter vaccine type
        vbrand_filter = (vacc['vaccine_doses']['brand'].data[:] == 2) | \
                        (vacc['vaccine_doses']['brand'].data[:] == 3)
        dvacc = dst.create_dataframe('vacc')
        vacc['vaccine_doses'].apply_filter(vbrand_filter, ddf=dvacc)

        #join asmt with vaccine using patient_id, write to result
        asmt_v = dst.create_dataframe('asmt_v')
        dataframe.merge(asmt, dvacc, asmt_v, 'patient_id', 'patient_id', how='inner')

        #filter healthy asmt record within 10days of vaccine date

        symp_list = ['persistent_cough', 'fever', 'fatigue', 'delirium', 'shortness_of_breath', 'diarrhoea',
                     'abdominal_pain', 'chest_pain', 'hoarse_voice', 'skipped_meals', 'loss_of_smell', 'headache',
                     'sore_throat', 'chills_or_shivers', 'eye_soreness', 'nausea', 'blisters_on_feet',
                     'unusual_muscle_pains', 'runny_nose', 'red_welts_on_face_or_lips', 'dizzy_light_headed',
                     'swollen_glands', 'sneezing', 'skin_burning', 'earache', 'altered_smell', 'brain_fog',
                     'irregular_heartbeat']
        symp_filter = asmt_v['persistent_cough'].data[:] > 1  # has symptom
        for symptom1 in symp_list:
            symp_filter |= asmt_v[symptom1].data[:] > 1  # has symptom
        symp_filter = ~symp_filter # has no symptom
        symp_filter &= asmt_v['date_taken_specific'].data[:] > asmt_v['updated_at_l'].data[:]  # asmt before vaccine
        symp_filter &= asmt_v['updated_at_l'].data[:] > asmt_v['date_taken_specific'].data[:] - 3600 * 24 * 10  # 10 days
        asmt_v.apply_filter(symp_filter)

        # has symptom after vaccine
        yes_symp_filter = asmt_v['persistent_cough'].data[:] > 1
        for symptom1 in symp_list:
            yes_symp_filter |= asmt_v[symptom1].data[:] > 1  # has symptom
        yes_symp_filter &= asmt_v['date_taken_specific'].data[:] < asmt_v['updated_at_l'].data[:]  # assessment after vaccine
        yes_symp_filter &= asmt_v['date_taken_specific'].data[:] + 3600 * 24 * 10 > asmt_v['updated_at_l'].data[:]  # assessment within 7 days of vaccine
        asmt_v.apply_filter(yes_symp_filter)
        print("finish asmt join vaccine.")

def join_tests():
    """
    Merge tests to previous merged (assessments, vaccine), filter out subjects has test records within 10days after vaccine
    """
    with sess.Session() as s:
        # open related datasets
        src = s.open_dataset(ADATA, 'r', 'asmt')
        tests_src = src['tests']
        dst = s.open_dataset(DSTDATA, 'r+', 'dst')
        vacc = dst['asmt_v']
        tests_m = dst.create_dataframe('tests_m')
        dataframe.merge(vacc, tests_src, tests_m, 'patient_id_l', 'patient_id', how='inner')

        # filter out subjects has tests after 10days of vaccine
        # date_taken_specific_l is vaccine date, date_taken_specific_r is tests date
        test_filter = tests_m['date_taken_specific_l'] < tests_m['date_taken_specific_r']  # test after vaccine
        test_filter &= tests_m['date_taken_specific_l'] > (tests_m['date_taken_specific_r'] - 3600 * 24 * 10)
        tests_m.apply_filter(test_filter)

def count():
    with sess.Session() as s:
        # open related datasets
        dst = s.open_dataset(DSTDATA, 'r', 'dst')
        vacc = dst['tests_m']
        print(len(dst['tests_m']['patient_id_l_l']))

if __name__ == '__main__':
    print(datetime.now())
    asmt_merge_vacc()
    join_tests()
    #count()
    print(datetime.now())
