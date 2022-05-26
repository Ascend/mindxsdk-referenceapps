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
import logging
import hashlib
import math
import time
import traceback
from merkle import verify_audit_proof
from server_config import LOG_INDEX_BY_USER_TABLE_NUMBER,\
    es_database_lock, secure_db_lock


def read_from_secure_db_for_statistic(user_id, secure_db_object): # 该函数从高安数据库获取该用户日志统计数据
    belonged_table_name = None
    logging.info('待查询用户为' + str(user_id))
    print('待查询用户为', user_id)
    if user_id != 'mindx': # 将user_id取模后获得所在的表序号
        divided_id = None
        try:
            divided_id = int(user_id, 16) % LOG_INDEX_BY_USER_TABLE_NUMBER
        except Exception: # 如果用户名不是可转为16进制的字符，则算一次哈希 除以16后向上取整，表示需要的字符数
            used_char_num = math.ceil(LOG_INDEX_BY_USER_TABLE_NUMBER / 16)
            divided_id = int(hashlib.sha256(user_id.encode()).hexdigest()[-used_char_num:], 16) % \
                LOG_INDEX_BY_USER_TABLE_NUMBER
        finally:
            belonged_table_name = 'log_index_by_user_' + str(divided_id).zfill(3)
    else:
        belonged_table_name = 'log_index_by_user_' + user_id
    logging.info('belonged_table_name' + str(belonged_table_name))
    print('belonged_table_name', belonged_table_name)
    secure_db_lock.acquire()
    statistic_result = secure_db_object.query_one_row(belonged_table_name, 'user_id', user_id)
    secure_db_lock.release()
    if statistic_result is not None:
        statistic_result_list = list(statistic_result)
        statistic_result_dict = {}
        if len(statistic_result_list) > 0:
            for i in statistic_result_list:
                statistic_result_dict[i[1]] = i[2]
        logging.info('statistic_result_dict' + str(statistic_result_dict))
        print('statistic_result_dict', statistic_result_dict)
        return True, statistic_result_dict
    else:
        logging.info('该用户在高安数据库中没有统计信息')
        print('该用户在高安数据库中没有统计信息')
        return False, {}


def read_from_es_once(es_object, i, query_json_time, query_json_id, expect_result_count):
    result_list = es_object.query_all(i, query_json_time, expect_result_count)
    result_count = len(result_list)
    logging.info('在ES库' + i + '中第1次查到' + str(result_count) + '条记录')
    print('在ES库', i, '中第1次查到', result_count, '条记录')
    if result_count < expect_result_count: # 如果在这个库中满足时间的条目并没有覆盖这个库里所有该用户的条目，再读一次，用于纠删
        temp_result = es_object.query_all(i, query_json_id, expect_result_count) # 注意query_json_id这里没有时间约束
        temp_result_count = len(temp_result) 
        if temp_result_count == expect_result_count:
            logging.info('在ES库' + str(i) + '中查到' + str(temp_result_count) + '条不分时间的记录，从条目上看ES库完整')
            print('在ES库', i, '中查到', temp_result_count, '条不分时间的记录，从条目上看ES库完整')
            return result_list, True
        else:
            logging.info('在ES库' + str(i) + '中查到' + str(temp_result_count) + '条不分时间的记录，从条目上看ES库不完整')
            print('在ES库', i, '中查到', temp_result_count, '条不分时间的记录，从条目上看ES库不完整')
            return result_list, False
    return result_list, True


def read_from_es_windows(es_object, i, query_json_time, query_json_id, expect_result_count):
    status_flag = True
    result_list = []
    read_count = math.ceil(expect_result_count / 10000)
    temp_list_first = es_object.query_all(i, query_json_time, 10000)
    query_json_time_with_sort = query_json_time
    if len(temp_list_first) >= 1:
        query_json_time_with_sort['search_after'] = temp_list_first[-1]['sort']
        result_list.extend(temp_list_first)
        logging.info('在ES库' + str(i) + '中第1次查到' + str(len(temp_list_first)) + '条记录')
        print('在ES库', i, '中第1次查到', len(temp_list_first), '条记录')
        for j in range(0, read_count - 1):
            temp_list = es_object.query_all(i, query_json_time_with_sort, 10000)
            if len(temp_list) >= 1:
                query_json_time_with_sort['search_after'] = temp_list[-1]['sort']
                result_list.extend(temp_list)
                logging.info('在ES库' + str(i) + '中第' + str(j + 2) + '次查到' + str(len(temp_list)) + '条记录')
                print('在ES库', i, '中第', j + 2, '次查到', len(temp_list), '条记录')
            else:
                break
    result_count = len(result_list)
    if result_count == expect_result_count:
        return result_list, status_flag
    else:
        status_flag = read_from_es_windows_time(es_object, i, query_json_id,
            read_count, expect_result_count)
        return result_list, status_flag


def read_from_es_windows_time(es_object, i, query_json_id, read_count, expect_result_count): 
    # 如果在这个库中满足时间的条目并没有覆盖这个库里所有该用户的条目，再读一轮，用于纠删
    result_list2 = []
    temp_list_first = es_object.query_all(i, query_json_id, 10000)
    query_json_id_with_sort = query_json_id
    if len(temp_list_first) >= 1:
        query_json_id_with_sort['search_after'] = temp_list_first[-1]['sort']
        result_list2.extend(temp_list_first)
        logging.info('在ES库' + i + '中第1次查到' + str(len(temp_list_first)) + '条不分时间的记录')
        print('在ES库', i, '中第1次查到', len(temp_list_first), '条不分时间的记录')
        for j in range(0, read_count):
            temp_list = es_object.query_all(i, query_json_id_with_sort, 10000)
            if len(temp_list) >= 1:
                query_json_id_with_sort['search_after'] = temp_list[-1]['sort']
                result_list2.extend(temp_list)
                logging.info('在ES库' + i + '中第' + str(j + 2) + '次查到' + str(len(temp_list)) + '条不分时间的记录')
                print('在ES库', i, '中第', j + 2, '次查到', len(temp_list), '条不分时间的记录')
            else:
                break
    result_count2 = len(result_list2)
    if result_count2 == expect_result_count:
        logging.info('在ES库' + str(i) + '中查到' + str(result_count2) + '条不分时间的记录，从条目上看ES库完整')
        print('在ES库', i, '中查到', result_count2, '条不分时间的记录，从条目上看ES库完整')
        return True
    else:
        logging.info('在ES库' + str(i) + '中查到' + str(result_count2) + '条不分时间的记录，从条目上看ES库不完整')
        print('在ES库', i, '中查到', result_count2, '条不分时间的记录，从条目上看ES库不完整')
        return False


def read_from_es(selected_user_id, start_time_str, end_time_str, statistic_result, es_object):
    query_json_id = {'sort':{'log_id':{'order':'asc'}}, 'query':{'match':{'item_user_id': selected_user_id}}}
    query_json_time = {'sort': {'log_id': {'order': 'asc'}},
        'query': {'bool': {'must':[{'match': {'item_user_id': selected_user_id}},
        {'range': {'item_time': {'lte': end_time_str, 'gte': start_time_str}}}]}}}
    es_results = []
    with es_database_lock:
        try:
            for i in list(statistic_result.keys()): 
                # i是es的index名；statistic_result[i]是该页面存储该用户的记录总数 
                # 注意这里把expect_result_count设置为记录总数仅为上限，满足时间约束的条目可能不够这个上限
                expect_result_count = statistic_result[i]
                result_list = []
                status_flag = True
                if expect_result_count <= 10000: # 小于安全门限，直接一次性读
                    result_list, status_flag = read_from_es_once(es_object, i, query_json_time,
                    query_json_id, expect_result_count)
                else: 
                    # 大于安全门限，需要用search_after的方式读
                    result_list, status_flag = read_from_es_windows(es_object, i, query_json_time,
                        query_json_id, expect_result_count)
                if status_flag is False:
                    return False, []
                es_results.extend(result_list)
            if 'search_after' in query_json_id:
                del query_json_id['search_after']
            if 'search_after' in query_json_time:
                del query_json_time['search_after']
        except Exception:
            logging.error(str(traceback.format_exc()))
            traceback.print_exc()
        finally:
            pass
    return True, es_results


def read_from_secure_db_for_roots(es_results, secure_db_object):
    log_id_list = [i['_source']['log_id'] for i in es_results]
    table_id_list = [i['_source']['log_index_mt_table_id'] for i in es_results]
    table_id_list = list(set(table_id_list)) # 去重，并转为list
    secure_db_lock.acquire()
    results = secure_db_object.query_multi_table_id('log_index_mt', table_id_list)
    secure_db_lock.release()
    if results is None or len(results) == 0: # 该起止时间未查到日志记录，退出
        logging.info('这些table_id未查到日志记录，退出')
        print('这些table_id未查到日志记录，退出')
        return False, [], {}
    used_root_list = [(i[0], i[2], i[3], i[6]) for i in list(results)] 
    # i[0] 是table id, i[2] [3] 是起止logid, i[6] 是root
    chunk_id_dict = {}
    used_item_list = []
    for i in used_root_list:
        chunk_id_dict[i[0]] = []
        for j in log_id_list: # 算法可以调优，这里有冗余计算
            if j >= i[1] and j <= i[2]:
                chunk_id_dict.get(i[0]).append(j)
                used_item_list.append((j, i[3]))
    return True, used_item_list, chunk_id_dict


def verify_audit_proofs(es_results, used_item_list):
    query_flag = True
    index_i = 0
    for es_item in es_results:
        tree_root = used_item_list[index_i][1]
        audit_proof_list = es_item['_source']['audit_proof']
        verified_result = None
        raw_content = es_item['_source']['item_raw_content']
        hash_leaf = hashlib.sha256(b'\x00' + raw_content.encode()).hexdigest()
        es_item['_source']['hash_value'] = hash_leaf
        if len(audit_proof_list) > 0: # 至少含有一个验证路径，说明是2叶以上造树的
            verified_result = verify_audit_proof(hash_leaf, audit_proof_list, tree_root)
        else: # 这是单条日志直接成树了，没有验证路径证据，直接比较哈希值和根
            print('第' + str(index_i + 1) + '条是单条日志直接成树了，没有验证路径证据，直接比较哈希值和根')
            if hash_leaf == tree_root:
                verified_result = True
            else:
                print('hash_leaf', hash_leaf, 'tree_root', tree_root)
                verified_result = False
        if verified_result is False:
            logging.info('第' + str(index_i + 1) + '条日志被篡改，对应log_id为' +
                  str(es_item['_source']['log_id']))
            print('第', index_i + 1, '条日志被篡改，对应log_id为',
                  es_item['_source']['log_id'])
            es_item['_source']['verify_result'] = 1
            query_flag = False
        else:
            es_item['_source']['verify_result'] = 0
        index_i += 1
    return query_flag


def verify_order(es_results, chunk_id_dict):
    es_result_index = 0
    for i in list(chunk_id_dict.keys()): # 这里i是chunk_id
        for j in range(0, len(chunk_id_dict[i])):
            if es_results[es_result_index]['_source']['one_user_order'] != j:
                logging.info('用户chunk内order数验证错误，第' + str(es_result_index) +\
                    '条记录出错，当前条目' + str(es_results[es_result_index]) + '应该为该用户在该chunk的第' + str(j) + '条')
                print('用户chunk内order数验证错误，第', es_result_index, '条记录出错，当前条目',\
                    es_results[es_result_index], '应该为该用户在该chunk的第', j, '条')
                return False
            es_result_index += 1
    return True

    
def user_search_main(es_object, secure_db_object1, secure_db_object2, selected_user_id, start_time_str, end_time_str):
    logging.info('待查询用户' + selected_user_id + '的日期范围为' + start_time_str + '至' + end_time_str)
    print('待查询用户', selected_user_id, '的日期范围为', start_time_str, '至', end_time_str)
    start_time_point_0 = time.time()  # 开始计时
    query_flag0, statistic_result = read_from_secure_db_for_statistic(selected_user_id, secure_db_object2) 
    # 该函数从高安数据库获取该用户日志的统计信息
    start_time_point_1 = time.time()
    time_differ = start_time_point_1 - start_time_point_0
    logging.info('用户审计第一阶段（数据预处理和高安数据库统计数据读取）耗时' + str(time_differ) + '秒')
    print('用户审计第一阶段（数据预处理和高安数据库统计数据读取）耗时', time_differ, '秒')
    if query_flag0 is False:
        return 4, []
    query_flag1, es_results = read_from_es(selected_user_id, start_time_str,\
        end_time_str, statistic_result, es_object) # 该函数从ES获取用户所有日志
    start_time_point_2 = time.time()
    time_differ = start_time_point_2 - start_time_point_1
    logging.info('用户审计第二阶段（ES数据读取）耗时' + str(time_differ) + '秒')
    print('用户审计第二阶段（ES数据读取）耗时', time_differ, '秒')
    if query_flag1 is False:
        return 5, []
    if query_flag1 is True and len(es_results) == 0:
        return 1, []
    query_flag2, used_item_list, chunk_id_dict = read_from_secure_db_for_roots(es_results, secure_db_object1) 
    # 该函数从高安数据库获取该用户日志相关的所有根
    start_time_point_3 = time.time()
    time_differ = start_time_point_3 - start_time_point_2
    logging.info('用户审计第三阶段（高安数据库根数据读取）耗时' + str(time_differ) + '秒')
    print('用户审计第三阶段（高安数据库根数据读取）耗时', time_differ, '秒')
    if query_flag2 is False:
        return 6, []
    query_flag3 = verify_audit_proofs(es_results, used_item_list) # 验证merkle树审计路径
    start_time_point_4 = time.time()
    time_differ = start_time_point_4 - start_time_point_3
    logging.info('用户审计第四阶段（Merkle树audit-proof验证）耗时' + str(time_differ) + '秒')
    print('用户审计第四阶段（Merkle树audit-proof验证）耗时', time_differ, '秒')
    if query_flag3 is False: # 逆转es_result，实现新结果在前
        es_results.reverse()
        return 2, es_results
    query_flag4 = verify_order(es_results, chunk_id_dict) # 验证单用户在每一chunk内的顺序
    start_time_point_5 = time.time()
    time_differ = start_time_point_5 - start_time_point_4
    logging.info('用户审计第五阶段（用户日志顺序验证）耗时' + str(time_differ) + '秒')
    print('用户审计第五阶段（用户日志顺序验证）耗时', time_differ, '秒')
    if query_flag4 is False:
        es_results.reverse() # 逆转es_result，实现新结果在前
        return 3, es_results
    start_time_point_6 = time.time()
    time_differ = start_time_point_6 - start_time_point_0
    logging.info('用户审计总计共耗时' + str(time_differ) + '秒')
    print('用户审计总计共耗时', time_differ, '秒')
    es_results.reverse() # 正常情况 逆转es_result，实现新结果在前
    return 0, es_results