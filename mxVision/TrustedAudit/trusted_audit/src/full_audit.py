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
import base64
import hashlib
import logging
from datetime import datetime
from merkle import MerkleTree
from server_config import LOG_INDEX_MT_NUMBER_THRESHOLD,\
    LOG_INDEX_MT_TIME_THRESHOLD, LOG_INDEX_MT_TABLE_ALL_KEYS,\
    LOG_INDEX_MT_TABLE_ALL_KEYS_APPENDIX, BLOCK_TABLE_ALL_KEYS,\
    BLOCK_TABLE_ALL_KEYS_APPENDIX, secure_db_lock


def create_tables_columns_for_full_audit(secure_db_object):
    table_name = 'log_index_mt'
    if secure_db_object.check_table_exists(table_name) is False:
        column_names_list = [i[0] + i[1] 
            for i in zip(LOG_INDEX_MT_TABLE_ALL_KEYS, LOG_INDEX_MT_TABLE_ALL_KEYS_APPENDIX)]
        secure_db_object.add_table_with_two_index(
            table_name, column_names_list)
    table_name = 'block'
    if secure_db_object.check_table_exists(table_name) is False:
        column_names_list = [
        i[0]+i[1] for i in zip(BLOCK_TABLE_ALL_KEYS, BLOCK_TABLE_ALL_KEYS_APPENDIX)]
        secure_db_object.add_table(table_name, column_names_list)
        keys_list = BLOCK_TABLE_ALL_KEYS[1:]
        temp_time = str(datetime.now())
        value_list = [temp_time, temp_time, temp_time, 'naive_tree_root',\
            'naive_prev_hash', 1, 0, 'block_signature']
        secure_db_object.add_item(table_name, keys_list, value_list)


def log_index_mt_waiting(log_index_mt_max_id, block_max_id, last_block_time, current_time):
    if log_index_mt_max_id // LOG_INDEX_MT_NUMBER_THRESHOLD > block_max_id - 1:
        return True # log_index_mt量够了，准备建大树
    else: # log_index_mt量不够，继续等
        if (current_time -
                last_block_time).seconds >= LOG_INDEX_MT_TIME_THRESHOLD:
            return True # log_index_mt等待时间够了，准备建block
        else:
            return False # log_index_mt等待时间不够，继续等
    return False


def process_and_write_log_index_mt_table(chunk_buffer, secure_db_object,
                                         es_index):
    len_chunk_buffer = len(chunk_buffer)
    sub_tree_start_time = chunk_buffer[0]['timestamp']  # 当前小树的最早时间
    sub_tree_end_time = chunk_buffer[-1]['timestamp']  # 当前小树的最晚时间
    hash_value_list = []
    for i in range(0, len_chunk_buffer):
        chunk_buffer[i]['item_raw_content'] = str(chunk_buffer[i])
        hash_value = hashlib.sha256(b'\x00' + chunk_buffer[i]['item_raw_content'].encode()).hexdigest()
        chunk_buffer[i]['hash_value'] = hash_value
        hash_value_list.append(hash_value)
    sub_tree = MerkleTree()
    sub_tree.insert_list(hash_value_list, True)
    sub_tree_root = sub_tree.root.hash # 取子树树根, hex格式, 64个字符的string
    for i in range(0, len_chunk_buffer):
        _, chunk_buffer[i]['audit_proof'] = sub_tree.get_proof_by_leaf_index(i)
    value_list_for_log_index_mt_table = [es_index, chunk_buffer[0]['log_id'],
        chunk_buffer[-1]['log_id'], sub_tree_start_time, sub_tree_end_time, sub_tree_root]
    secure_db_lock.acquire()
    secure_db_object.add_item('log_index_mt', LOG_INDEX_MT_TABLE_ALL_KEYS[1:],\
        value_list_for_log_index_mt_table) # 'table_id'定义为自增，不需要操作
    secure_db_lock.release()


def process_and_decide_to_write_block_table(current_time, secure_db_object):
    # 该函数读取log_index_mt表的总量、block表的上一块，判断是否需要block表写入
    last_item_in_log_index_mt = secure_db_object.query_max_id(
        'log_index_mt', 'table_id')
    if isinstance(last_item_in_log_index_mt, list) and len(last_item_in_log_index_mt) >= 0: # 防止读空的log_index_mt表
        log_index_mt_max_id = last_item_in_log_index_mt[0][0]
    last_item_in_block = secure_db_object.query_max_row('block', 'table_id')
    last_block = None
    if isinstance(last_item_in_block, list) and len(last_item_in_block) >= 0: # 防止读空的log_index_mt表
        last_block = last_item_in_block[0]
    block_max_id = last_block[0]
    last_block_time = last_block[3]
    if log_index_mt_waiting(log_index_mt_max_id, block_max_id,\
        last_block_time, current_time):
        return True, last_block, log_index_mt_max_id
    else:
        return False, None, log_index_mt_max_id


def write_to_block_table(current_time, last_block, secure_db_object):
    block_max_id = last_block[0]
    last_tree_start_leaf_id = last_block[6]
    current_tree_start_leaf_id = last_tree_start_leaf_id + last_block[7] # last_block[7] 是上一个block中包含多少个小树根
    secure_db_lock.acquire() # 注意，读回来的时间是datetime格式
    log_items = secure_db_object.query_multi_row_one_predict('log_index_mt',\
            'table_id', current_tree_start_leaf_id)
    if log_items is None:
        logging.info('出现没有读到log_mt表的错误，再尝试一次')
        print('出现没有读到log_mt表的错误，再尝试一次')
        log_items = secure_db_object.query_multi_row_one_predict(
            'log_index_mt', 'table_id', current_tree_start_leaf_id)
    secure_db_lock.release()
    if log_items is None:
        logging.info('第二次仍旧没有读到log_mt表的错误')
        print('第二次仍旧没有读到log_mt表的错误')
        return
    log_items_list = list(log_items)
    whole_tree_start_time = str(log_items_list[0][4]) # 最早未写入大树叶子的开始时间
    items_roots_list = [i[6] for i in log_items_list]
    main_tree = MerkleTree()
    main_tree.insert_list(items_roots_list, False)
    main_tree_root = main_tree.root.hash # 取子树树根, hex格式, 64个字符的string
    prev_hash = hashlib.sha256(str(last_block).encode()).hexdigest()
    whole_tree_end_time = str(log_items_list[-1][4]) # 最晚未写入大树叶子的结束时间
    value_list_for_block_table_temp = [block_max_id + 1,\
        whole_tree_start_time, whole_tree_end_time,\
        str(current_time), main_tree_root, prev_hash,\
            current_tree_start_leaf_id, len(items_roots_list)]
    signature = 'NULL' # 未来需要外部审计时需要签名，此处预留空位
    value_list_for_block_table = value_list_for_block_table_temp[1:]
    value_list_for_block_table.append(signature)
    secure_db_lock.acquire()
    secure_db_object.add_item('block', BLOCK_TABLE_ALL_KEYS[1:],
                          value_list_for_block_table)
    secure_db_lock.release()

    
def log_processing_for_full_audit(chunk_buffer, secure_db_obj, es_index):
    process_and_write_log_index_mt_table(chunk_buffer, secure_db_obj, es_index)
    current_time = datetime.now()
    write_to_block_table_flag, last_block, log_index_mt_max_table_id\
        = process_and_decide_to_write_block_table(current_time, secure_db_obj)
    if write_to_block_table_flag is True:
        write_to_block_table(current_time, last_block, secure_db_obj)
    return log_index_mt_max_table_id