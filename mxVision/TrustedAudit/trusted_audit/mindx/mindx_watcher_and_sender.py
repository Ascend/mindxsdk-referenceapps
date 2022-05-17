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
import traceback
import threading
import time
import os
from queue import Queue
from datetime import datetime
import psutil
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


def sending_time_threshold_check(last_item_timestamp):
    global SENDING_GAP_TIME
    last_time = time.mktime(time.strptime(last_item_timestamp[0:19], '%Y-%m-%d %H:%M:%S'))
    if time.time() - last_time >= SENDING_GAP_TIME:
        return True
    else:
        return False


def handle_file(f1, start_line, year_info):
    output = []
    for count, content in enumerate(f1, 1):
        if count < start_line:
            continue
        try: # content 是读取的一行没有格式信息的纯文本信息；这里对content格式化处理
            if content.startswith('Log file created at:')\
            or content.startswith('Running on machine:')\
            or content.startswith('Log line format:'): # 如果是开头的三行，不处理
                continue
            content_split = content.split(']')
            if len(content_split) <= 1: # 空行 不处理
                print('warning:FILE_NAME', FILE_NAME, 'fail:line', count, '是空行，内容为', content)
                continue
            first_part = content_split[0]
            msg = ' '.join(content_split[1:]) # msg是第一个]符号后的所有内容
            first_part_split = first_part.split(' ')
            if len(content_split) <= 1: # 无时间戳 不处理
                print('warning:FILE_NAME', FILE_NAME, 'fail:line', count, '无时间戳，内容为', content)
                continue
            month_day = first_part_split[0][1:]
            hour_min_second_micro = first_part_split[1]
            thread_id = first_part_split[2]
            file_line = first_part_split[3]
            # IWEF标识符，表示Information/debug Warning Error Fault;
            # 用年信息加上日志条目中的20位月日时分秒信息，精确到微秒;
            # 最终格式为 yyyymmdd hh:mm:ss.uuuuuu 的字符串;5位的thread_id;
            #形如file:line的file_line;msg信息
            temp_content_dict = {'type': content[0],
                'timestamp': year_info + '-' + month_day[0:2] + '-' + month_day[2:] + ' ' + hour_min_second_micro,
                'thread_id': thread_id,  'file_line': file_line,  'msg': msg}
            output.append(temp_content_dict)
        except Exception:
            traceback.print_exc()
            print('error: FILE_NAME', FILE_NAME, 'fail: line', count, '行读取错误，内容为content', content)
            continue
        finally:
            pass
    return output


def open_file(filename, start_line, year_info):
    output = []
    with open(filename) as f1:
        output = handle_file(f1, start_line, year_info)
    print('info: FILE_NAME', FILE_NAME, '本次读取', len(output), '行')
    return output


def read_file(filename, log_file_info): # 从start_line的行号往后读取多行
    start_line = log_file_info[0]
    year_info = log_file_info[1]
    count = None
    output = []
    try:
        output = open_file(filename, start_line, year_info)
    except Exception:
        traceback.print_exc()
        print('error: FILE_NAME', FILE_NAME, '文件打开失败，该文件可能被自动删除')
        return output
    finally:
        pass
    return output


def log_sender(): # 声明sending_buffer为全局变量
    global SENDING_BUFFER, SENDING_QUEUE, SENDING_CHUNK_SIZE
    while True:
        _ = SENDING_QUEUE.get() 
        # 双门限：1.如果发缓冲区里已经有足够多的日志条目，则读取后发送;
        # 2.如果发缓冲区里的最后一条时间字符串转化为UTC（精确到秒）超过当前时间的5秒，也发送
        sending_buffer_lock.acquire()
        url = 'http://172.18.0.4:1234/TransparentLog_mindx'
        send_flag = False
        if len(SENDING_BUFFER) >= SENDING_CHUNK_SIZE:
            print(datetime.now(), '时发送', len(SENDING_BUFFER), '条日志，因为item_count_threshold')
            send_flag = True
        elif len(SENDING_BUFFER) > 0 and  sending_time_threshold_check(SENDING_BUFFER[-1]['timestamp']):
            print(datetime.now(), '时发送', len(SENDING_BUFFER), '条日志，因为time_threshold')
            send_flag = True
        if send_flag is True:
            try:
                rsp = requests.request('POST', url, json=SENDING_BUFFER)
                print('发送结果', rsp, rsp.text)
                SENDING_BUFFER.clear()
                SENDING_BUFFER = []
            except Exception:
                traceback.print_exc()
                time.sleep(5)
            finally:
                pass
        sending_buffer_lock.release()


class MyFileSystemEventHandler(FileSystemEventHandler):

    def on_created(self, event):
        global FILE_NAME, LOG_FILES
        FILE_NAME = event.key[1].replace(folder_name, '', 1)[1:] 
        # 创建新文件时触发, 文件名为把路径删除掉的字符串，再删除开头的斜杠
        if FILE_NAME.startswith('mxsdk.log.'):
            print(datetime.now(), '时检测到新建文件', FILE_NAME) 
            # 列表的第一个是存储读取行标志的，从1开始; 列表的第二个是存储4位的年信息
            year_info = FILE_NAME.split('.')[3][0:4]
            LOG_FILES[FILE_NAME] = [1, year_info]
        else: # 临时文件一般以.开头或数字文件名，不处理
            pass

    def on_modified(self, event): # 文件修改时触发
        global LOG_FILES, FILE_NAME, SENDING_BUFFER 
        if event.key[1].endswith('/logs'): # 如果事件结尾是/logs则说明是目录级的变动，不处理
            pass
        else: # 文件名为把路径删除掉的字符串，再删除开头的斜杠
            FILE_NAME = event.key[1].replace(folder_name, '', 1)[1:]
            if FILE_NAME.startswith('mxsdk.log.'):
                if FILE_NAME not in LOG_FILES:
                    year_info = FILE_NAME.split('.')[3][0:4] # 取4位的年信息
                    LOG_FILES[FILE_NAME] = [1, year_info]
                target_line = LOG_FILES.get(FILE_NAME)
                temp_lines = read_file(event.key[1], target_line)
                if isinstance(temp_lines, list) and len(temp_lines) > 0: # 对于行数并未改变的行为，或者文件读取出错，不处理
                    LOG_FILES.get(FILE_NAME)[0] += len(temp_lines)
                    sending_buffer_lock.acquire()
                    SENDING_BUFFER.extend(temp_lines)
                    SENDING_QUEUE.put('hello')
                    print(datetime.now(), '时检测到文件', FILE_NAME, '变化，当前已读取到', LOG_FILES.get(FILE_NAME)[0], '行')
                    sending_buffer_lock.release()
    
    def on_deleted(self, event):
        global LOG_FILES, FILE_NAME
        if event.key[1].endswith('/logs'):  # 如果事件结尾是/logs则说明是目录级的变动，不处理
            pass
        else: # 文件名为把路径删除掉的字符串，再删除开头的斜杠
            if FILE_NAME.startswith('mxsdk.log.'):
                print(datetime.now(), '时检测到文件', FILE_NAME, '删除')
                if FILE_NAME in LOG_FILES:
                    del LOG_FILES[FILE_NAME]
                print(datetime.now(), '从LOG_FILES中删除', FILE_NAME, '字段')


def process_check(processname):
    count = 0
    pl = psutil.pids()
    for pid in pl:
        try:
            if (psutil.Process(pid).exe().startswith('/usr/local/bin/python') or \
            psutil.Process(pid).exe().startswith('/usr/bin/python') or \
            psutil.Process(pid).exe().startswith('python')) and \
            len(psutil.Process(pid).cmdline()) >= 3 and \
            psutil.Process(pid).cmdline()[2] == processname:
                count += 1
        except psutil.NoSuchProcess:
            continue
        finally:
            pass
    if count > 1: # 有1个是正常的，有2个说明已经有1个启动了，这个不启动
        return True
    else:
        return False


if __name__ == "__main__":
    FOLDER_NAME = '/work/mindx_sdk/mxVision/logs'
    LOG_FILES = {} # 全局变量LOG_FILES用于监控各文件日期、已读取行号等信息
    SENDING_CHUNK_SIZE = 100 # 发缓冲区大小；buffer大小建议设为chunk大小的若干倍，避免阻塞
    SENDING_GAP_TIME = 5
    SENDING_BUFFER = []
    SENDING_QUEUE = Queue()
    sending_buffer_lock = threading.Lock()
    # 本程序由绝对路径启动
    watcher_file_path = os.path.join(sys.path[0], 'mindx_watcher_and_sender.py')
    print('watcher_file_path', watcher_file_path)
    if process_check(watcher_file_path) is True:
        print('watcher已经启动，不再启动watcher，退出')
        sys.exit()
    print('watcher is running...datetime is:', datetime.now())
    sender_thread = threading.Thread(target=log_sender) # 发送进程
    sender_thread.start()
    event_handler = MyFileSystemEventHandler() # 文件监控进程
    observer = Observer()
    observer.schedule(event_handler, path=folder_name, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    finally:
        pass
    observer.join()