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
import collections
import traceback
import logging
import json
import os
import socket
import sys
import time
import threading
from datetime import datetime
from queue import Queue
from elasticsearch import helpers
from flask import Flask, request
from flask_cors import CORS
from flask_restful import reqparse
import objsize
from es_database_operate import ESDatebase
from full_audit import log_processing_for_full_audit
from server_config import ES_MAP_SCHEMA, CHUNK_ITEM_NUMBER_THRESHOLD, CHUNK_TIME_THRESHOLD,\
    CHUNK_ITEM_NUMBER_THRESHOLD_UP, ES_INDEX_THRESHOLD, ES_PRE_INDEX, es_database_lock, SEARCH_BUFFER_SIZE
from database_init import gauss_database_init, obtain_largest_log_id
from user_audit import log_processing_for_user_audit
from user_search import user_search_main
from full_search import full_search_main
app = Flask(__name__)
CORS(app, supports_credentials=True)


def write_to_es_process(one_chunk, log_index_mt_table_id, es_index):
    es_to_be_write = []
    for i in one_chunk:
        temp_to_write_dict_content = {
            'log_id': i['log_id'], 'item_type': i['item_type'],  'item_user_id': i['user_id'],
            'item_time': i['timestamp'],  'one_user_order': i['one_user_order'],
            'item_raw_content': i['item_raw_content'],
            'audit_proof': i['audit_proof'], 'log_index_mt_table_id': log_index_mt_table_id
        }
        temp_to_write_total_dict = {'_op_type': 'index', '_index': es_index, '_source': temp_to_write_dict_content }
        es_to_be_write.append(temp_to_write_total_dict)
    return es_to_be_write


def write_to_es_handle(es_object, es_to_be_write):
    for ok, response in helpers.parallel_bulk(es_object.obj_es, es_to_be_write):
        if not ok:
            logging.info('es写入错误' + response)
            print('es写入错误' + response)
        continue


def write_to_es(one_chunk, es_object, es_index, log_index_mt_table_id):
    es_to_be_write = []
    es_to_be_write = write_to_es_process(one_chunk, log_index_mt_table_id, es_index)
    for i in range(0, 5):
        try:
            es_database_lock.acquire()
            write_to_es_handle(es_object, es_to_be_write)
            if (i != 0):
                print('日志es写入第二阶段错误过，第' + str(i + 1) + '次尝试成功')
                logging.info('日志es写入第二阶段错误过，第' + str(i + 1) + '次尝试成功')
            break
        except Exception:
            logging.error(str(traceback.format_exc()))
            logging.error('日志es写入第二阶段，第' + str(i + 1) + '次尝试错误')
            traceback.print_exc()
            print('日志es写入第二阶段错误，第' + str(i + 1) + '次尝试错误')
            time.sleep(20)
        finally:
            es_database_lock.release()


def es_index_check(es_object, global_es_id, current_es_index_list):
    index_number = global_es_id // ES_INDEX_THRESHOLD + 1
    if index_number > len(current_es_index_list): # 把上一个index的字符串最后一个字符加1
        temp_index_string = ES_PRE_INDEX + str(
            int(current_es_index_list[-1][0][-3:]) + 1).zfill(3)
        with es_database_lock:
            try:
                es_object.add_index(temp_index_string, ES_MAP_SCHEMA)
                temp_index_tuple = (temp_index_string, global_es_id)
                current_es_index_list.append(temp_index_tuple) # 函数直接改变ES_INDEX_LIST的值
            except Exception:
                logging.error(str(traceback.format_exc()))
                traceback.print_exc()
            finally:
                pass


def set_timer():
    global TIMER_START_FLAG
    global CHUNK_BUFFER
    logging.info('定时器' + str(CHUNK_TIME_THRESHOLD) + '秒等待时间够了，准备处理')
    print('定时器', CHUNK_TIME_THRESHOLD, '秒等待时间够了，准备处理')
    chunk_buffer_lock.acquire()
    process_token_queue.put(CHUNK_BUFFER)
    CHUNK_BUFFER = []
    chunk_buffer_lock.release()
    TIMER_START_FLAG = False


def split_list_by_n_sublist(input_list, n):
    for i in range(0, len(input_list), n):
        yield input_list[i:i+n]


def read_queue():
    global TIMER_START_FLAG
    global TIMER_THREAD
    global CHUNK_BUFFER
    if len(CHUNK_BUFFER) >= CHUNK_ITEM_NUMBER_THRESHOLD: # 如果小于上限，至少处理一次chunk，所以把定时器关闭
        if TIMER_START_FLAG is True: # 如果定时器开启，则关闭
            logging.info('定时器已开启，关闭')
            print('定时器已开启，关闭')
            TIMER_THREAD.cancel()
            TIMER_THREAD = None
            TIMER_START_FLAG = False
        temp_list = None
        if len(CHUNK_BUFFER) > CHUNK_ITEM_NUMBER_THRESHOLD_UP: # 如果大于上限，按上限大小为chunk分割处理
            temp_list = split_list_by_n_sublist(CHUNK_BUFFER, CHUNK_ITEM_NUMBER_THRESHOLD_UP)
            for i in temp_list:
                process_token_queue.put(i)
        else: # 如果小于上限，按原始大小为chunk直接处理
            temp_list = CHUNK_BUFFER
            process_token_queue.put(temp_list)
        CHUNK_BUFFER = []
    else:
        if TIMER_START_FLAG is False: # 如果定时器关闭，则开启
            logging.info('定时器已关闭，开启' + str(CHUNK_TIME_THRESHOLD) + '秒定时器')
            print('定时器已关闭，开启', CHUNK_TIME_THRESHOLD, '秒定时器')
            TIMER_START_FLAG = True
            TIMER_THREAD = threading.Timer(CHUNK_TIME_THRESHOLD, set_timer)
            TIMER_THREAD.start()


def auditor(es_object, secure_db_object1, secure_db_object2):
    global NEXT_LOG_ID
    global ES_INDEX_LIST
    chunk_index = 1
    global_es_id = max(ES_INDEX_LIST[-1][1], NEXT_LOG_ID)
    while True:
        chunk_item = process_token_queue.get()
        len_chunk_item = len(chunk_item)
        if len_chunk_item == 0:
            continue
        time_point_1 = datetime.now()
        log_info = '准备处理第%s个chunk，含%s条日志，当前时间%s' %\
                   (str(chunk_index), str(len_chunk_item), str(time_point_1))
        logging.info(log_info)
        print(log_info)
        try:
            chunk_item.sort(key=lambda a:datetime.strptime(a['timestamp'],\
                '%Y-%m-%d %H:%M:%S.%f'))
            for item in chunk_item:
                item['log_id'] = global_es_id
                global_es_id += 1
            log_processing_for_user_audit(chunk_item, ES_INDEX_LIST[-1][0],\
                secure_db_object2) # user_audit 逻辑 (需要先于full_audit跑，用于添加one_user_order)
            logging.info('第' + str(chunk_index) + '个chunk的user_audit 逻辑完成')
            log_index_mt_max_table_id = log_processing_for_full_audit(\
                chunk_item, secure_db_object1, ES_INDEX_LIST[-1][0]) # full_audit 逻辑, 加入es_index, 方便之后查找
            logging.info('第' + str(chunk_index) + '个chunk的full_audit 逻辑完成')
            write_to_es(chunk_item, es_object, ES_INDEX_LIST[-1][0],\
                        log_index_mt_max_table_id)
            logging.info('第' + str(chunk_index) + '个chunk的write_to_es 逻辑完成')
            es_index_check(es_object, global_es_id, ES_INDEX_LIST)
            logging.info('第' + str(chunk_index) + '个chunk的es_index_check 逻辑完成')
            NEXT_LOG_ID = global_es_id
        except Exception:
            logging.error(str(traceback.format_exc()))
            traceback.print_exc()
        finally:
            pass
        time_point_2 = datetime.now()
        log_info = '处理完第%s个chunk，含%s条日志，当前时间%s耗时%s' % (str(chunk_index),\
            str(len_chunk_item), str(time_point_2), str(time_point_2 - time_point_1))
        logging.info(log_info)
        print(log_info)
        chunk_index += 1


@app.route('/TransparentLog_gateway', methods=['POST'])
def transparent_log_gateway():
    content = request.json
    if not isinstance(content, list) or not isinstance(content[0], dict):
        return 'updated items are invalid', 501
    timepoint = str(datetime.now())
    str_len_content = str(len(content))
    logging.info('收到落库请求，包含' + str_len_content + '行网关日志， 写入文件队列，时间：' + str(timepoint))
    print('收到落库请求，包含', str_len_content, '行网关日志， 写入文件队列，时间：' + str(timepoint))
    index_i = 0 # 网关数据只能精确到秒，这里手动补上微秒信息
    for item in content:
        if 'timestamp' in item:
            if len(item['timestamp']) == 19:
                item['timestamp'] += '.000000'
        else:
            logging.info('该请求的第' + str(index_i) + '行网关日志没有时间戳！！！')
            print('该请求的第' + str(index_i) + '行网关日志没有时间戳！！！')
        item['item_type'] = 1
        index_i += 1
    chunk_buffer_lock.acquire()
    CHUNK_BUFFER.extend(content)
    read_queue()
    chunk_buffer_lock.release()
    resp = 'upload ' + str_len_content + ' items success when ' + timepoint
    return resp, 201


@app.route('/TransparentLog_mindx', methods=['POST'])
def transparent_log_mindx():
    content = request.json
    if not isinstance(content, list) or not isinstance(content[0], dict):
        return 'updated items are invalid', 501
    timepoint = str(datetime.now())
    str_len_content = str(len(content))
    log_info = '收到落库请求，包含' + str_len_content + '行mindx日志，写入文件队列，时间：' + str(timepoint)
    logging.info(log_info)
    print(log_info)
    for i in content:
        i['user_id'] = 'mindx'
        i['item_type'] = 3
    chunk_buffer_lock.acquire()
    CHUNK_BUFFER.extend(content)
    read_queue()
    chunk_buffer_lock.release()
    resp = 'upload ' + str_len_content + ' items success when ' + timepoint
    return resp, 200


def process_return_window(search_result, req):
    if len(search_result) > (req['page_num'] - 1) * req['page_size']:
        if len(search_result) < req['page_size']: # 查询结果比窗口小，把所有结果都返回
            return [i['_source'] for i in search_result]
        else: # 查询结果比窗口大或相等
            temp_items = search_result[(req['page_num'] - 1) * req['page_size']: req['page_num'] * req['page_size']]
            return [i['_source'] for i in temp_items]
    else: # 请求过大的页面，已经不包含条目了
        return []


@app.route('/Searcher', methods=['POST'])
def searcher():
    global NEXT_LOG_ID, SEARCH_BUFFER
    req = request.json
    logging.info('收到查找请求：' + str(req) + '当前的NEXT_LOG_ID为' + str(NEXT_LOG_ID))
    print('收到查找请求：', req, '当前的NEXT_LOG_ID为', NEXT_LOG_ID)
    search_result = None
    err_code = None
    resp = {'order_info':None}
    searcher_lock.acquire()
    if (req['user_id'], req['start_time'], req['end_time'], NEXT_LOG_ID) in SEARCH_BUFFER: 
        # 之前搜索过同样的结果，则可以直接返回已经查询过的结果；
        # NEXT_LOG_ID用于标记数据库在查询间隙没有改动, err_code = -1 表示之前查询过的值
        err_code = -1
        search_result = SEARCH_BUFFER[(req['user_id'], req['start_time'], req['end_time'], NEXT_LOG_ID)]
    else:
        # 调用搜索接口; err_code为返回值错误情况；result为查询结果;
        # err_code = 0 是表示查询结果全部正确, err_code = 1 表示在请求的时间段内未查询到任何数据,
        # err_code = 2 表示发现了某条审计路径验证错误, err_code = 3 表示发现了某条日志在chunk块内的顺序验证错误,
        # err_code = 4 表示高安库里无统计信息,err_code = 5 表示es里条目数和统计信息有差异，库被删了,
        # err_code = 6 表示没有读到高安库里的根
        err_code, search_result = user_search_main(g_es_object, g_secure_db_object1,
            g_secure_db_object2, req['user_id'], req['start_time'], req['end_time'])
        if len(SEARCH_BUFFER.keys()) >= SEARCH_BUFFER_SIZE:
            SEARCH_BUFFER.popitem(last=False)
        SEARCH_BUFFER[(req['user_id'], req['start_time'], req['end_time'], NEXT_LOG_ID)] = search_result
    searcher_lock.release()
    resp = {'total_len': len(search_result)}
    resp['content'] = process_return_window(search_result, req)
    resp['err_code'] = err_code
    logging.info('err_code' + str(err_code) + 'len(result)' + str(len(search_result)))
    print('err_code', err_code, 'len(result)', len(search_result))
    return json.dumps(resp), 200


@app.route('/Searcher_full', methods=['POST'])
def searcher_full():
    global NEXT_LOG_ID, SEARCH_BUFFER
    req = request.json
    logging.info('收到查找请求：' + str(req) + '当前的NEXT_LOG_ID为' + str(NEXT_LOG_ID))
    print('收到查找请求：', req, '当前的NEXT_LOG_ID为', NEXT_LOG_ID)
    search_result = None
    err_code = None
    resp = {'order_info':None}
    searcher_lock.acquire()
    if (req['start_time'], req['end_time'], NEXT_LOG_ID) in SEARCH_BUFFER: 
        # 之前搜索过同样的结果，则可以直接返回已经查询过的结果； 
        # NEXT_LOG_ID用于标记数据库在查询间隙没有改动, err_code = -1 表示之前查询过的值
        err_code = -1
        search_result = SEARCH_BUFFER[ (req['start_time'], req['end_time'], NEXT_LOG_ID)]
    else: 
        # 调用搜索接口; err_code为返回值错误情况；result为查询结果,
        # err_code = 0 是表示查询结果全部正确, err_code = 1 表示在请求的时间段内未查询到任何数据, 
        # err_code = 2 是表示查询结果不全，es库被删了, err_code = 3 表示重构根不正确
        err_code, search_result = full_search_main(g_es_object, g_secure_db_object1, req['start_time'], req['end_time'])
        if len(SEARCH_BUFFER.keys()) >= SEARCH_BUFFER_SIZE:
            SEARCH_BUFFER.popitem(last=False)
        SEARCH_BUFFER[ (req['start_time'], req['end_time'], NEXT_LOG_ID) ] = search_result
    searcher_lock.release()
    resp = {'total_len': len(search_result)}
    resp['content'] = process_return_window(search_result, req)
    resp['err_code'] = err_code
    logging.info('err_code' + str(err_code) + 'len(result)' + str(len(search_result)))
    print('err_code', err_code, 'len(result)', len(search_result))
    return json.dumps(resp), 200


def port_check(input_ip, input_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((input_ip, int(input_port)))
        s.shutdown(2)
        return True
    except Exception:
        return False
    finally:
        pass


if __name__ == '__main__':
    parser = reqparse.RequestParser()
    parser.add_argument('content')
    process_token_queue = Queue(5) # 用于读取日志到日志总处理的chunk队列
    CHUNK_BUFFER = []
    TIMER_START_FLAG = False
    TIMER_THREAD = None
    chunk_buffer_lock = threading.Lock()
    searcher_lock = threading.Lock()
    g_es_object = ESDatebase()
    time.sleep(5)
    g_secure_db_object1, g_secure_db_object2 = gauss_database_init('database1', 'database2')
    SEARCH_BUFFER = collections.OrderedDict()
    ES_INDEX_LIST = []
    NEXT_LOG_ID = obtain_largest_log_id(g_secure_db_object1, ES_INDEX_LIST)
    cur_dir = os.path.abspath(__file__).rsplit('/', 1)[0]
    log_path = os.path.join(cur_dir, 'server.log')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('根据高安数据库，下一个log_id为' + str(NEXT_LOG_ID))
    print('根据高安数据库，下一个log_id为', NEXT_LOG_ID)
    PORT = 1234
    IP_ADDRESS = '0.0.0.0'
    if port_check(IP_ADDRESS, PORT):
        logging.info(IP_ADDRESS + ':' + str(PORT) + '已被占用，请检查后重试，退出。')
        print(IP_ADDRESS + ':' + str(PORT) + '已被占用，请检查后重试，退出。')
        sys.exit()
    logging.info('可信日志服务端, docker制作日期2022年5月5日')
    print('可信日志服务端, docker制作日期2021年5月5日')
    auditor_thread = threading.Thread(target=auditor, args=(g_es_object, g_secure_db_object1, g_secure_db_object2))
    auditor_thread.start()
    app.run(port=PORT, host=IP_ADDRESS, threaded=True)