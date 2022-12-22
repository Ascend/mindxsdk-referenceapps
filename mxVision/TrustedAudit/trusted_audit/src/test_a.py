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
import random
import os
import requests


def test_gateway_query(item_number):
    name_ids = os.urandom(16).hex()
    print('mock日志含1个用户，共计', item_number, '条日志')
    one_second = timedelta(seconds=1)
    current_time = datetime.now()
    item_list = []
    for i in range(0, item_number):
        current_time += one_second
        user_id = name_ids
        item_dict = {'client_ip': 'xxx.xxx.xxx.xx', 'request_uri': 'a' * 10,
            'host_ip': 'xxx.xxx.xxx.xx', 'route_id': '1', 'timestamp': str(current_time),
            'user_id': user_id, 'query_id': 'a' * 32}
        item_list.append(item_dict)
    print('模仿网关生成mock的用户名和起止时间为', name_ids, item_list[0]['timestamp'], item_list[-1]['timestamp'])
    rsp = requests.request('POST', 'http://xxx.xxx.xxx.xx:xxxx/TransparentLog_gateway', json = item_list)
    print(rsp.text)


if __name__ == '__main__':
    test_gateway_query(25)