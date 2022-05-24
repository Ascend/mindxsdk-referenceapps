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
from datetime import timedelta, datetime
import sys
import requests


def search_query_user(user_id, start_time, end_time, page_num, page_size):
    search_query_msg = {'user_id': user_id, 'start_time': start_time, 
        'end_time': end_time, 'page_num':page_num, 'page_size':page_size}
    url = 'http://172.18.0.4:1234/Searcher'
    search_query(search_query_msg, url)


def search_query(search_query_msg, url):
    rsp = requests.request('POST', url, json = search_query_msg).json()
    all_item_count = rsp['total_len']
    rsp_items = rsp['content']
    if rsp_items is not None:
        log_ids = [ i['log_id'] for i in rsp_items]
        verified_result_list = [ i['verify_result'] for i in rsp_items ]
        print('检索到', all_item_count, '条日志，第', search_query_msg['page_num'], '页的',
            search_query_msg['page_size'], '条日志对应的log_id为', log_ids, '验证结果为', verified_result_list)
    else:
        print('检索到', all_item_count, '条日志，第', search_query_msg['page_num'], '页不含日志')


if __name__ == '__main__':
    if len(sys.argv) < 8:
        print('error: 输入参数错误，退出')
    search_query_user(sys.argv[1], sys.argv[2] + ' ' + sys.argv[3],
        sys.argv[4] + ' ' + sys.argv[5], int(sys.argv[6]), int(sys.argv[7]))