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
import threading
# 高安数据库口令
GAUSS_ROOT_USER = 'gaussdb'
GAUSS_ROOT_PWD = 'Enmo@123'
GAUSS_DB_NAME = 'mindxlog'
GAUSS_USER = 'myauditor'
GAUSS_PWD = 'Huawei123'
# ES相关变量
ES_INDEX_THRESHOLD = 10000000
ES_SAFE_QUERY_THRESHOLD = 10000
ES_PRE_INDEX = 'es_index_'
ES_MAP_SCHEMA = {
    'mappings': {
        'properties': {
            'log_id': {
                'type': 'long'
            },
            'item_type': {
                'type': 'byte'
            },
            'item_user_id': {
                'type': 'text'
            },
            'item_time': {
                'type': 'date',
                'format': 'yyyy-MM-dd HH:mm:ss.SSSSSS'
            },
            'one_user_order': {
                'type': 'integer'
            },
            'item_raw_content': {
                'type': 'text'
            },
            'audit_proof': {
                'type': 'text'
            },
            'log_index_mt_table_id': {
                'type': 'long'
            },
        }
    },
    'settings': {
        'index': {
            'max_result_window': ES_SAFE_QUERY_THRESHOLD
        }
    }
}
# 字段名称: log_id 类型: long 含义与作用: 用于标识log_id，长整型，最大到2^63-1
# 字段名称: item_type 类型: byte 含义与作用: 用于标识日志条目的种类，当前支持来自于网关、AI模型、攻击检测三类日志；根据端口来源分类: 网关: 0x01；AI模型: 0x02；攻击检测: 0x03。
# 字段名称: item_user_id 类型: text 含义与作用: 用于标识用户id，来自于网关的日志条目中可以提取该字段；AI模型日志指定user ID为ffffffff；攻击检测日志指定userID为fffffffe。
# 字段名称: item_time 类型: date 格式 yyyy-MM-dd HH:mm:ss 含义与作用: 用于标识该日志条目记录内容所产生的时间
# 字段名称: item_raw_content 类型: text 含义与作用: 用于存储原始日志内容。
# 全盘审计相关常量
LOG_INDEX_MT_TABLE_ALL_KEYS = ['table_id', 'es_index_name', 'start_log_id',
    'end_log_id', 'start_time', 'end_time', 'items_root']
LOG_INDEX_MT_TABLE_ALL_KEYS_APPENDIX = [' BIGSERIAL', ' CHAR(12) NULL', ' BIGINT NULL',
    ' BIGINT NULL', ' timestamp(6) without time zone NULL', ' timestamp(6) without time zone NULL', ' CHAR(64) NULL']
# 字段名称: table_id 类型: INT UNSIGNED AUTO_INCREMENT 含义与作用: 作为主键由日志服务器分配，自增，且连续，从1开始
# 字段名称: es_index_name 类型: CHAR(12) 存储12个字符的字符串，不可变长度 含义与作用: es表对应存储的index名，形如es_index_000
# 字段名称: start_log_id 类型: BIGINT UNSIGNED 含义与作用: 代表ES库中所存储的日志条目开始位置。
# 字段名称: end_log_id 类型: BIGINT UNSIGNED 含义与作用: ES库中所需要连续读取的日志条目结束为止。
# 字段名称: start_time 类型: DATETIME 含义与作用: 用于标识生成该记录的时间，该时间与实际log生成时间稍有延迟，但可以通过与上一条目该字段进行运算，大概定位日志产生时间。
# 字段名称: end_time 类型: DATETIME 含义与作用: 用于标识生成该记录的时间，该时间与实际log生成时间稍有延迟，但可以通过与上一条目该字段进行运算，大概定位日志产生时间。
# 字段名称: items_root 类型: CHAR(64) 存储32B的哈希，不可变长度 含义与作用: 该字段为以日志条目杂凑为叶子节点计算MT生成的MTR
#   所有叶子节点信息可以通过start_log_id和item_count从ES数据库中获取
BLOCK_TABLE_ALL_KEYS = ['table_id', 'start_time', 'end_time', 'block_time', 'mt_mtr', 'prev_hash',
    'start_mt_id', 'mt_nums', 'block_signature']
BLOCK_TABLE_ALL_KEYS_APPENDIX = [' BIGSERIAL', ' timestamp(6) without time zone NULL',
    ' timestamp(6) without time zone NULL', ' timestamp(6) without time zone NULL', ' CHAR(64) NULL',
    ' CHAR(64) NULL', ' INT NULL', ' INT NULL', ' CHAR(128) NULL']
# 字段名称: table_id 类型: INT UNSIGNED AUTO_INCREMENT 含义与作用: 作为主键由日志服务器分配，自增，且连续。
# 字段名称: start_time 类型: DATETIME 含义与作用: 生成该条记录中MT_mtr字段的第一叶子节点对应的start_time。
# 字段名称: end_time 类型: DATETIME 含义与作用: 生成该条记录中MT_mtr字段的最后叶子节点对应的end_time。
# 字段名称: block_time 类型: DATETIME 含义与作用: 生成该条记录的时间
# 字段名称: mt_mtr 类型: CHAR(64) 存储32B的哈希，不可变长度 含义与作用: Log_index_MT table中Items_root字段为叶子，创建默克尔树所生成的默克尔根
# 字段名称: prev_hash 类型: CHAR(64) 存储32B的哈希，不可变长度 含义与作用: 上一条Block记录信息的杂凑值，用于后项和前项之间形成逻辑的链式关系。
# 字段名称: start_mt_id 类型: INT UNSIGNED 含义与作用: 该字段数值为Log_index_MT table中table_id主键，
#   表示以该table_id记录对应的items_root字段的内容开始创建MT，从table_id往下（增大）的MT_nums条记录所对应的
#   items_root字段都将是该MT的叶子节点，MT的根对应在MT_mtr字段。
# 字段名称: mt_nums 类型: INT UNSIGNED 含义与作用: 该字段数值为创建MT_mtr字段对应的MT所使用的叶子节点的个数。
# 字段名称: block_signature 类型: CHAR(128) 存储32+32B的ECDSA签名，含义与作用: 该字段数值为该条记录的start_time,
#   end_time, MT_mtr, prev_hash, start_MT_id, MT_nums键值的签名信息。
CHUNK_ITEM_NUMBER_THRESHOLD = 500  # 建小树的叶子门限下限
CHUNK_ITEM_NUMBER_THRESHOLD_UP = 2000  # 建小树的叶子门限上限
CHUNK_TIME_THRESHOLD = 10  # 建小树的时间门限，单位秒
LOG_INDEX_MT_NUMBER_THRESHOLD = 500  # 建大树的叶子门限
LOG_INDEX_MT_TIME_THRESHOLD = 500  # 建大树的时间门限，单位秒
# 用户审计相关常量
LOG_INDEX_BY_USER_TABLE_ALL_KEYS = ['user_id', 'es_index_name', 'user_cnt']
LOG_INDEX_BY_USER_TABLE_ALL_KEYS_APPENDIX = [' CHAR(32) NULL', ' VARCHAR(32) NULL', ' BIGINT NULL']
# 字段名称: user_id 类型: CHAR(32) 存储16B的user_id，不可变长度 含义与作用: 表示该条记录由哪个user产生。
# 字段名称: user_id 类型: CHAR(32) 存储16B的es_index_name，不可变长度 含义与作用: 表示该条记录在es中存在哪个es_index产生。
# 字段名称: user_cnt 类型: INT UNSIGNED 含义与作用: 用于标识该条记录为该用户的第几条日志条目记录。每个用户的记录都单增。从1开始
LOG_INDEX_BY_USER_TABLE_NUMBER = 16
es_database_lock = threading.Lock()
secure_db_lock = threading.Lock()