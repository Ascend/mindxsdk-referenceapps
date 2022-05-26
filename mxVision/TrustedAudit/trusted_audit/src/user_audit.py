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
import numpy
from server_config import LOG_INDEX_BY_USER_TABLE_ALL_KEYS, LOG_INDEX_BY_USER_TABLE_ALL_KEYS_APPENDIX,\
    LOG_INDEX_BY_USER_TABLE_NUMBER, secure_db_lock


def create_total_table_for_user_audit(secure_db_object):
    logging.info('正在建log_index_by_user表')
    print('正在建log_index_by_user表')
    for j in range(0, LOG_INDEX_BY_USER_TABLE_NUMBER):
        table_name = 'log_index_by_user_%s' % (str(j).zfill(3))
        if secure_db_object.check_table_exists(table_name) is False:
            column_names_list = [i[0] + i[1] for i in zip(LOG_INDEX_BY_USER_TABLE_ALL_KEYS,\
                LOG_INDEX_BY_USER_TABLE_ALL_KEYS_APPENDIX)]
            secure_db_object.add_table_with_index_without_pkey(table_name, column_names_list)
    logging.info('建' + str(LOG_INDEX_BY_USER_TABLE_NUMBER) + '张总表：log_index_by_user_000至' +\
        str(LOG_INDEX_BY_USER_TABLE_NUMBER - 1))
    print('建', LOG_INDEX_BY_USER_TABLE_NUMBER, '张总表：log_index_by_user_000至',
          LOG_INDEX_BY_USER_TABLE_NUMBER - 1)
    if secure_db_object.check_table_exists('log_index_by_user_mindx') is False:
        column_names_list = [i[0] + i[1] for i in zip(LOG_INDEX_BY_USER_TABLE_ALL_KEYS,
            LOG_INDEX_BY_USER_TABLE_ALL_KEYS_APPENDIX)]
        secure_db_object.add_table_with_index_without_pkey('log_index_by_user_mindx', column_names_list)


def pre_process_for_current_chunk(raw_log_items):
    # 该函数将相关用户按模操作分为若干表，与高安数据库中对应,
    # 输出结果例如 {'log_index_by_user_000':{'Alice':1,'Bob':2},'log_index_by_user_001':{'Carol':3,'David':4}}
    user_list = [i['user_id'] for i in raw_log_items] # 提取用户列表
    statistic_result = dict(zip(*numpy.unique(user_list, return_counts=True))) 
    # 统计用户出现次数，结果为用户名-次数的字典，结果形如{'Alice':49,'Carol':51}
    related_users_count_in_current_chunk = {}
    for i in list(statistic_result.keys()):  # i是用户id的str型
        if i != 'mindx': # 确定用户i在哪一张高安数据库的统计表中
            divided_i = None
            try:
                divided_i = int(i, 16) % LOG_INDEX_BY_USER_TABLE_NUMBER
            except Exception: 
                # 如果用户名不是可转为16进制的字符，则算一次哈希 除以16后向上取整，表示需要的字符数
                used_char_number = math.ceil(LOG_INDEX_BY_USER_TABLE_NUMBER / 16) 
                divided_i = int(hashlib.sha256(i.encode()).hexdigest()[-used_char_number:], 16) % \
                    LOG_INDEX_BY_USER_TABLE_NUMBER
            finally:
                pass
            belonged_table_name = 'log_index_by_user_' + str(divided_i).zfill(3)
        else:
            belonged_table_name = 'log_index_by_user_' + i
        if belonged_table_name not in related_users_count_in_current_chunk: 
            # 如果这个表还没有分配，分配给一个空字典
            related_users_count_in_current_chunk[belonged_table_name] = {}
        related_users_count_in_current_chunk.get(belonged_table_name)[i] = statistic_result.get(i)
        one_user_items = [j for j in raw_log_items if j['user_id'] == i] 
        # 这里不需要再按log_id排序，因为上一步提取就是顺序提取
        last_order = 0 # 为某一用户的所有条目增加序号
        for j in range(0, statistic_result[i]):
            one_user_items[j]['one_user_order'] = last_order
            last_order += 1
    return related_users_count_in_current_chunk


def total_table_write(related_users_count_in_current_chunk,\
    es_index_name, secure_db_object): # 对即将写入大房间的数据进行处理并批量写入
    secure_db_lock.acquire()
    for i in related_users_count_in_current_chunk.items(): 
        # i[0]是表名；list(i[1].keys())是用户名的列表；list(i[1].values())是对应用户名出现次数的列表
        secure_db_object.update_user_cnt(i[0], list(i[1].keys()),
            es_index_name, list(i[1].values()))
    secure_db_lock.release()

    
def log_processing_for_user_audit(raw_log_items, es_index_name, secure_db_object):# 该函数将相关用户按模操作分为若干表
    related_users_count_in_current_chunk = pre_process_for_current_chunk(
        raw_log_items) # 统计每个用户有多少条日志
    total_table_write(related_users_count_in_current_chunk, es_index_name,
                      secure_db_object) # 把统计信息、es_index_name写入高安数据库