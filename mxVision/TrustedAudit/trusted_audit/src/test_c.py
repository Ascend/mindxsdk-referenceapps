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
from test_b import search_query

    
def search_query_full(start_time, end_time, page_num, page_size):
    search_query_msg = {'start_time': start_time, 'end_time': end_time,
        'page_num':page_num, 'page_size':page_size}
    url = 'http://172.18.0.4:1234/Searcher_full'
    search_query(search_query_msg, url)


if __name__ == '__main__':
    if len(sys.argv) < 7:
        print('error: 输入参数错误，退出')
    search_query_full(sys.argv[1] + ' ' + sys.argv[2],
        sys.argv[3] + ' ' + sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))