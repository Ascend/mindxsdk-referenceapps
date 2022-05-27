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
import os
from datetime import datetime
import psutil


def process_check(processname):
    count = 0
    pl = psutil.pids()
    kill_count = 0
    for pid in pl:
        try:
            if (psutil.Process(pid).exe().startswith('/usr/local/bin/python') or \
            psutil.Process(pid).exe().startswith('/usr/bin/python') or \
            psutil.Process(pid).exe().startswith('python')) and \
            len(psutil.Process(pid).cmdline()) >= 3 and \
            psutil.Process(pid).cmdline()[2] == processname:
                psutil.Process(pid).kill()
                kill_count += 1
        except psutil.NoSuchProcess:
            continue
        finally:
            pass
    if kill_count > 0:
        print('杀了', kill_count, '个watcher进程')
        return True
    else:
        return False
if __name__ == "__main__":
    watcher_file_path = os.path.join(
        sys.path[0], 'mindx_watcher_and_sender.py')
    print(datetime.now(), 'watcher_file_path', watcher_file_path)
    if process_check(watcher_file_path) is True:
        print(datetime.now(), 'watcher已经启动，停止watcher进程，退出')
        sys.exit()
    print(datetime.now(), 'watcher未启动，退出')

