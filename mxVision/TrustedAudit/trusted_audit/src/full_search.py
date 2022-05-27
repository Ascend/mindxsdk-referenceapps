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
import hashlib
import logging
import math
import time
import traceback
from datetime import datetime
from merkle import MerkleTree
from server_config import es_database_lock, secure_db_lock


def read_log_index_mt(start_time_str, end_time_str, secure_db):
    secure_db_lock.acquire() # 被查询对象包含在查询窗口中
    items_result = secure_db.query_window('log_index_mt',
        'end_time', start_time_str, 'start_time', end_time_str)
    secure_db_lock.release()
    if items_result is None or len(items_result) == 0:
        logging.info('该起止时间未查到日志记录')
        print('该起止时间未查到日志记录')
        return []
    else:
        return list(items_result)


def read_from_es_default(start_log_id, end_log_id, es_idx_name, es_object):
    query_js = {'sort': {'log_id': {'order': 'asc'}},
        'query': {'bool': {'must': {'range': {
            'log_id': {'gte': start_log_id, 'lte': end_log_id}}}}}}
    expect_count = end_log_id - start_log_id + 1
    result_list = []
    result_count = 0
    es_database_lock.acquire()
    print('准备读', es_idx_name, '起止logid', start_log_id, end_log_id, '共', expect_count, '条')
    if expect_count <= 10000:
        result_list = es_object.query_all(es_idx_name, query_js, expect_count)
        result_count = len(result_list)
        logging.info('在ES库' + es_idx_name + '中第1次查到' + str(result_count) + '条记录')
        print('在ES库', es_idx_name, '中第1次查到', result_count, '条记录')
    else:
        read_count = math.ceil(expect_count / 10000)
        temp_list = es_object.query_all(es_idx_name, query_js, 10000)
        query_js_with_sort = query_js
        if len(temp_list) >= 1:
            query_js_with_sort['search_after'] = temp_list[-1]['sort']
            result_list.extend(temp_list)
            logging.info('在ES库' + es_idx_name + '中第1次查到' + str(len(temp_list)) + '条记录')
            print('在ES库', es_idx_name, '中第1次查到', len(temp_list), '条记录')
        for i in range(0, read_count - 1):
            print('准备第', i + 2, '次读')
            temp_list = es_object.query_all(es_idx_name, query_js_with_sort, 10000)
            if len(temp_list) >= 1:
                query_js_with_sort['search_after'] = temp_list[-1]['sort']
                result_list.extend(temp_list)
                logging.info('在ES库' + es_idx_name + '中第' + str(i+2) + '次查到' + str(len(temp_list)) + '条记录')
                print('在ES', es_idx_name, '第', i+2, '次查到', len(temp_list), '条')
            else:
                break
        result_count = len(result_list)
    es_database_lock.release()
    if 'search_after' in query_js:
        del query_js['search_after']
    if isinstance(result_list, list):
        logging.info('在ES库' + es_idx_name + '中查到' + str(result_count) + '条记录，应当有' + str(expect_count) + '条记录')
        print('在ES库', es_idx_name, '中查到', result_count, '条记录，应当有',\
              expect_count, '条记录')
        if result_count < expect_count: # 数目不够够，说明丢失；注意chunk内条目不会跨es_index
            return False, []
    return True, result_list


def step_two_result_process(query_items):
    related_es_index_dict = {} 
    # related_es_index_dict是字典的字典，第一级字典key是es_index，第二级字典key是log_id, value是起止log_id的
    es_results_dict = {} 
    # es_results_dict 是字典的字典，第一级字典key是es_index，第二级字典key是log_id，value是es读回来的日志列表
    searched_root_dict = {} 
    # searched_root_dict 是字典的字典，第一级字典key是es_index，第二级字典key是log_id，value是高安读回来的树根
    for i in query_items:
        if i[1] not in related_es_index_dict:
            related_es_index_dict[i[1]] = {i[0]: ([i[2], i[3]])}
        else:
            related_es_index_dict.get(i[1])[i[0]] = ([i[2], i[3]])
        if i[1] not in es_results_dict:
            searched_root_dict[i[1]] = {i[0]: i[6]}
        else:
            searched_root_dict.get(i[1])[i[0]] = i[6]
        if i[1] not in es_results_dict:
            es_results_dict[i[1]] = {i[0]: []}
        else:
            es_results_dict.get(i[1])[i[0]] = []
    return related_es_index_dict, es_results_dict, searched_root_dict


def step_three_one_root_mark(mark_flag, input_dict, start_time_dt,\
                             end_time_dt):
    es_results_list = []
    for k in input_dict: # es_results_dict[i][j]
        k['_source']['verify_result'] = mark_flag
        temp_timestamp = datetime.strptime(
            k['_source']['item_time'], '%Y-%m-%d %H:%M:%S.%f') # 只留存在时间窗口内的日志条目
        if start_time_dt <= temp_timestamp <= end_time_dt:
            es_results_list.append(k)
    return es_results_list


def step_three_check_root(start_time_str, end_time_str, es_results_dict, searched_root_dict):
    es_results_list = []
    error_flag = 0
    start_time_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
    end_time_dt = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S.%f')
    for i in list(es_results_dict.keys()): # i是index名，j是table_id
        for j in list(es_results_dict[i].keys()):
            temp_leaves = [k['_source']['item_raw_content'] for k in es_results_dict[i][j]]
            temp_leaves_hash = [hashlib.sha256(b'\x00' + k.encode()).hexdigest() for k in temp_leaves]
            sub_tree = MerkleTree()
            sub_tree.insert_list(temp_leaves_hash, True)
            temp_root = sub_tree.root.hash
            if temp_root == searched_root_dict[i][j]:
                es_results_list.extend(step_three_one_root_mark(0, es_results_dict[i][j], start_time_dt, end_time_dt))
            else: # 重构根不对
                es_results_list.extend(step_three_one_root_mark(2, es_results_dict[i][j], start_time_dt, end_time_dt))
                error_flag = 3
    return error_flag, es_results_list


def full_search_main(es_object, secure_db, start_time_str, end_time_str):
    time_point_1 = time.time()
    query_items = read_log_index_mt(start_time_str, end_time_str, secure_db) # 用起止时间直接从log_index_mt中定位大树叶子-小树根
    if len(query_items) == 0:
        return 1, []
    current_time = time.time()
    logging.info('全盘审计第一阶段（高安数据库）数据读取' + str(len(query_items)) +\
        '条数据耗时' + str(current_time - time_point_1) + '秒')
    print('全盘审计第一阶段（高安数据库）数据读取', len(query_items), '条数据耗时', current_time - time_point_1, '秒')
    related_es_index_dict, es_results_dict, searched_root_dict = step_two_result_process(query_items)
    time_point_2 = time.time()
    for i in list(related_es_index_dict.keys()): # i是index名，j是table_id
        for j in list(related_es_index_dict.get(i).keys()):
            query_flag1, temp_es_results = read_from_es_default(
                related_es_index_dict.get(i).get(j)[0], related_es_index_dict.get(i).get(j)[1], i, es_object)
            if query_flag1 is False:
                logging.info('es库被删了')
                print('es库被删了')
                return 2, []
            else:
                es_results_dict.get(i)[j] = temp_es_results
    current_time = time.time()
    logging.info('全盘审计第二阶段（ES数据读取）耗时' + str(current_time - time_point_2) + '秒')
    print('全盘审计第二阶段（ES数据读取）耗时', current_time - time_point_2, '秒')
    time_point_3 = time.time()
    error_flag, es_results = step_three_check_root(
        start_time_str, end_time_str, es_results_dict, searched_root_dict) # 重构树，验证根
    current_time = time.time()
    logging.info('全盘审计第三阶段（重构根）耗时' + str(current_time - time_point_3) + '秒')
    logging.info('全盘审计总计耗时' + str(current_time - time_point_1) + '秒')
    print('全盘审计第三阶段（重构根）耗时', current_time - time_point_3, '秒')
    print('全盘审计总计耗时', current_time - time_point_1, '秒')
    es_results.reverse()
    # es_results里的对象都是可以直接引用修改的， 所以重构根后增加的验证结果是可以直接对原对象改变的，
    # 这里直接输出 最终结果再排序，且是按log_id降序排序，实现新结果在前
    return error_flag, es_results