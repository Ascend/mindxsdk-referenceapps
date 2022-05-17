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
import sys
from es_database_operate import ESDatebase
from user_search import read_from_es, read_from_secure_db_for_statistic
from server_config import ES_PRE_INDEX, GAUSS_ROOT_USER, GAUSS_ROOT_PWD, GAUSS_DB_NAME
from gauss_database_operate import GaussDatabase
from database_init import gauss_database_init


if __name__ == '__main__':
    selected_user_id = sys.argv[1]
    modified_content = sys.argv[2]
    START_TIME_STR = '2021-06-10 00:00:00.000000'
    END_TIME_STR = '2023-06-23 00:00:00.000000'
    es_object = ESDatebase()
    secure_db_object1, secure_db_object2 = gauss_database_init('database1', 'database2')
    result_flag, statistic_result = read_from_secure_db_for_statistic(selected_user_id, secure_db_object2)
    query_flag1, es_results = read_from_es(selected_user_id, START_TIME_STR,
        END_TIME_STR, statistic_result, es_object)
    es_results.reverse()
    SELECTED_ITEM = 0
    if len(es_results) >= 5:
        SELECTED_ITEM = 5
    elif len(es_results) < 5:
        SELECTED_ITEM = len(es_results) - 1
    temp_dict = es_results[SELECTED_ITEM]['_source']
    MODIFIED_TARGET = str({'key':modified_content})
    temp_dict['item_raw_content'] = MODIFIED_TARGET
    es_object.add_item(index_name = es_results[SELECTED_ITEM]['_index'],
        global_id = es_results[SELECTED_ITEM]['_id'], dict_item_obj = temp_dict)
    print('log_id为', es_results[SELECTED_ITEM]['_source']['log_id'],
    '的ES数据中item_raw_content字段已被修改为', MODIFIED_TARGET)