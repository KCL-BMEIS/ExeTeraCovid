from datetime import datetime
import numpy as np

import exetera.core.session as sess
from exetera.core import dataframe
from exetera.core import operations as ops
from exetera.core.abstract_types import Field


# ADATA = '/home/jd21/data/processed_May17_processed.hdf5'
# VDATA = '/home/jd21/data/vacc.0603.h5'
# DSTDATA = '/home/jd21/data/full_merge.h5'
ADATA = '/mnt/timemachine_marta/processed_May17_processed.hdf5'
VDATA = '/mnt/data/jd21/data/vacc.0603.h5'
DSTDATA = 'merge.h5'

def get_aggregate_index(index_fld, target_fld, how='max'):
    if isinstance(index_fld, list):
        spans = ops._get_spans_for_2_fields_by_spans(index_fld[0].get_spans(), index_fld[1].get_spans())
    elif isinstance(index_fld, Field):
        spans = index_fld.get_spans()

    result_idx = np.zeros(len(spans)-1, dtype='int')
    for i in range(0, len(spans)-1):
        if spans[i+1] - spans[i] == 1:
            result_idx[i] = spans[i]
        else:
            if how == 'max':
                result_idx[i] = spans[i] + np.argmax(target_fld.data[spans[i]:spans[i+1]])
            elif how == 'min':
                result_idx[i] = spans[i] + np.argmin(target_fld.data[spans[i]:spans[i+1]])
            elif how == 'first':
                result_idx[i] = spans[i]
            elif how == 'last':
                result_idx[i] = spans[i+1] - 1
    return result_idx

def SymptomJoinTests():
    begin = datetime.strptime("2020-12-08", '%Y-%m-%d').timestamp()
    end = datetime.strptime("2021-05-17", '%Y-%m-%d').timestamp()
    with sess.Session() as s:
        # open related datasets
        src = s.open_dataset(ADATA, 'r', 'src')
        asmt = src['assessments']
        tests = src['tests']
        dst = s.open_dataset(DSTDATA, 'w', 'dst')
        dst_asmt = dst.create_dataframe('dst_asmt')

        #copy asmt
        dst_asmt['patient_id'] = asmt['patient_id']
        dst_asmt['updated_at'] = asmt['updated_at']

        #filter asmt
        filter = asmt['updated_at'] >= begin
        filter &= asmt['updated_at'] <= end
        nhs_criteria = asmt['loss_of_smell'] == 2
        nhs_criteria |= asmt['fever'] == 2
        nhs_criteria |= asmt['persistent_cough'] == 2
        filter &= nhs_criteria
        #apply filter
        dst_asmt.apply_filter(filter)
        print('number of unique patient for 1) ', len(dst_asmt['patient_id'].get_spans())-1)

        #join tests
        dst_tests = dst.create_dataframe('dst_tests')
        dataframe.merge(dst_asmt, tests, dest=dst_tests, left_on='patient_id', right_on='patient_id', how='inner')

        # filter tests againt symp updated_at_l symp date     updated_at_r tests date
        filter = dst_tests['updated_at_l'] < dst_tests['updated_at_r']
        filter &= dst_tests['updated_at_l'] + 10 * 24 * 3600 > dst_tests['updated_at_r']
        dst_tests.apply_filter(filter)
        print('number of unique patient for 2) ', len(dst_tests['patient_id_l'].get_spans()) - 1)

        aggidx = get_aggregate_index(dst_tests['patient_id_l'],  dst_tests['result'], how='max')
        print('unique tested positive: ', np.sum(np.where(dst_tests['result'].data[aggidx] == 4, True, False)))


if __name__ == '__main__':
    print(datetime.now())
    SymptomJoinTests()
    print(datetime.now())