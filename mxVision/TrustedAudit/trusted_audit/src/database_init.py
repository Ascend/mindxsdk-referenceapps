# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from es_database_operate import ESDatebase
from full_audit import create_tables_columns_for_full_audit
from server_config import ES_PRE_INDEX, ES_MAP_SCHEMA,\
    ES_INDEX_THRESHOLD, GAUSS_ROOT_USER, GAUSS_ROOT_PWD, GAUSS_DB_NAME
from user_audit import create_total_table_for_user_audit
from gauss_database_operate import GaussDatabase


def database_init_delete(es_object, secure_db_object1, secure_db_object2):
    all_es_index_list = es_object.get_index_of_all()
    for i in all_es_index_list:
        es_object.delete_index(i)
    if secure_db_object1.check_table_exists('block'):
        secure_db_object1.delete_table('block', True)
    if secure_db_object1.check_table_exists('log_index_mt'):
        secure_db_object1.delete_table('log_index_mt', True)
    if secure_db_object2.check_table_exists('log_index_by_user_mindspore'):
        secure_db_object2.delete_table('log_index_by_user_mindspore', True)
    if secure_db_object2.check_table_exists('freq_user_mt_mindspore'):
        secure_db_object2.delete_table('freq_user_mt_mindspore', True)
    for i in secure_db_object2.all_tables():
        if i.startswith('log_index_by_user_') or i.startswith(
                'freq_user_mt_'):
            secure_db_object2.delete_table(i, False)
    print('旧表删除完成')


def gauss_database_init(db_name1, db_name2):
    gauss_sys_object = GaussDatabase(username=GAUSS_ROOT_USER,\
        password=GAUSS_ROOT_PWD, db_name=GAUSS_DB_NAME)
    gauss_sys_object.check_user_and_add_user_authority()
    gauss_auditor_object = GaussDatabase()
    if not gauss_auditor_object.check_database_exists(db_name1):
        gauss_auditor_object.create_datebase(db_name1)
    if not gauss_auditor_object.check_database_exists(db_name2):
        gauss_auditor_object.create_datebase(db_name2)
    gauss_auditor_object1 = GaussDatabase(db_name=db_name1)
    gauss_auditor_object2 = GaussDatabase(db_name=db_name2)
    return gauss_auditor_object1, gauss_auditor_object2


def obtain_largest_log_id(secure_db_object1, es_index_list):
    query = secure_db_object1.query_log_id_max()
    temp_next_log_id = None
    if query is None or len(query) == 0:
        temp_next_log_id = 0
    else:
        temp_next_log_id = query[0][3] + 1
    es_index_count = temp_next_log_id // ES_INDEX_THRESHOLD + 1
    for i in range(0, es_index_count):
        es_index_list.append(
            (ES_PRE_INDEX + str(i).zfill(3), i*ES_INDEX_THRESHOLD))
    return temp_next_log_id


if __name__ == '__main__':
    es_object_test = ESDatebase() 
    secure_db_object1_test, secure_db_object2_test = gauss_database_init('database1', 'database2')
    database_init_delete(es_object_test, secure_db_object1_test, secure_db_object2_test)
    es_object_test.add_index(ES_PRE_INDEX + '000', ES_MAP_SCHEMA)
    create_tables_columns_for_full_audit(secure_db_object1_test)
    create_total_table_for_user_audit(secure_db_object2_test)