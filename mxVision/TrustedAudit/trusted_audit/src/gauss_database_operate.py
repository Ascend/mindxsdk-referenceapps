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
import re
import traceback
import psycopg2
from server_config import GAUSS_ROOT_USER, GAUSS_ROOT_PWD, GAUSS_USER, GAUSS_PWD, GAUSS_DB_NAME


class GaussDatabase():
    GAUSS_IP = '172.18.0.11'
    GAUSS_DB_PORT = 5432

    def __init__(self, host_ip=GAUSS_IP, username=GAUSS_USER,\
        password=GAUSS_PWD, db_name=GAUSS_DB_NAME, port_number=GAUSS_DB_PORT):
        self.connect_info = {'host':host_ip, 'user':username,  'passwd':password, 'db':db_name, 'port':port_number}

    def connect(self):
        return psycopg2.connect(host=self.connect_info.get('host'), user=self.connect_info.get('user'),
            password=self.connect_info.get('passwd'), database=self.connect_info.get('db'),
            port=self.connect_info.get('port'))
    
    def check_user_and_add_user_authority(self, user_name=GAUSS_USER, pwd=GAUSS_PWD):
        sql_select = 'SELECT * FROM pg_user'
        exist_flag = False
        connection = None
        cursor = None
        try:
            connection = self.connect()
            cursor = connection.cursor()
            cursor.execute(sql_select)
            rs = cursor.fetchall()
            rs_list_format = [i[0] for i in rs]
            if user_name in rs_list_format:
                exist_flag = True
                print('用户', user_name, '已存在')
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            print('高斯DB，判断用户是否存在错误')
        finally:
            if connection:
                cursor.close()
                connection.close()
        if exist_flag is True:
            return
        print('用户不存在，创建用户', user_name)
        sql = 'CREATE USER ' + user_name + ' PASSWORD ' + ' \'' + pwd + '\';'
        try:
            connection = self.connect()
            cursor = connection.cursor()
            cursor.execute(sql)
            connection.commit()
            print('创建用户', user_name, '成功')
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            print('创建用户', user_name, '失败')
        finally:
            if connection:
                cursor.close()
                connection.close()
        self.add_authority(user_name)
    
    def add_authority(self, user_name):
        sql = 'grant all privileges to ' + user_name + ';' # 用户授权1
        try:
            connection = self.connect()
            cursor = connection.cursor()
            cursor.execute(sql)
            connection.commit()
            print('用户', user_name, '授权Sysadmin成功')
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            print('用户', user_name, '授权Sysadmin失败')
        finally:
            if connection:
                cursor.close()
                connection.close()
        sql = 'ALTER ROLE ' + user_name + ' createdb;' # 用户授权2
        try:
            connection = self.connect()
            cursor = connection.cursor()
            cursor.execute(sql)
            connection.commit()
            print('用户', user_name, '授权createdb成功')
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            print('用户', user_name, '授权createdb失败')
        finally:
            if connection:
                cursor.close()
                connection.close()
    
    def check_database_exists(self, database_name_to_check):
        sql_select = 'SELECT datname FROM pg_database'
        connection = None
        cursor = None
        rs_list_format = []
        try:
            connection = self.connect()
            cursor = connection.cursor()
            cursor.execute(sql_select)
            rs = cursor.fetchall()
            rs_list_format = [i[0] for i in rs]
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            print('高斯DB，判断库是否存在错误')
        finally:
            if connection:
                cursor.close()
                connection.close()
        if database_name_to_check in rs_list_format:
            return True
        else:
            return False
    
    def create_datebase(self, database_name):
        sql = 'CREATE DATABASE ' + database_name + ';'
        connection = None
        cursor = None
        try:
            connection = self.connect()
            cursor = connection.cursor()
            connection.set_isolation_level(0)
            cursor.execute(sql)
            connection.set_isolation_level(1)
            connection.commit()
            print('创建数据库', database_name, '成功')
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            print('创建数据库', database_name, '失败')
        finally:
            if connection:
                cursor.close()
                connection.close()
    
    def all_tables(self):
        sql_show = "SELECT distinct(tablename) FROM pg_tables "
        sql_show += "WHERE SCHEMANAME = \'public\';"
        connection = None
        cursor = None
        try:
            connection = self.connect()
            cursor = connection.cursor()
            cursor.execute(sql_show)
            table_list = [tuple[0] for tuple in cursor.fetchall()]
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            print('高斯DB，查询所有表名错误')
        finally:
            if connection:
                cursor.close()
                connection.close()
        return table_list
    
    def check_table_exists(self, table_name_to_check):
        sql_show = 'SELECT distinct(tablename) FROM pg_tables'
        sql_show += ' WHERE SCHEMANAME = \'public\';'
        table_list_from_sql = []
        connection = None
        cursor = None
        try:
            connection = self.connect()
            cursor = connection.cursor()
            cursor.execute(sql_show)
            tables = [cursor.fetchall()]
            table_list_from_sql = re.findall('(\'.*?\')', str(tables))
            table_list_from_sql = [
                re.sub("'", '', each) for each in table_list_from_sql
            ]
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            print('高斯DB，查询表', table_name_to_check, '是否存在错误')
        finally:
            if connection:
                cursor.close()
                connection.close()
        if table_name_to_check in table_list_from_sql:
            return True
        else:
            return False
    
    def delete_table(self, table_name, debug=False):
        sql = 'DROP TABLE ' + table_name + ' ;'
        if self.check_table_exists(table_name) is False:
            print('表', table_name, '不存在')
            return
        else:
            connection = None
            cursor = None
            try:
                connection = self.connect()
                cursor = connection.cursor()
                cursor.execute(sql)
                connection.commit()
                print('删除表', table_name, '成功')
            except (Exception, psycopg2.Error) as error:
                traceback.print_exc()
                print('删除表', table_name, '失败')
            finally:
                if connection:
                    cursor.close()
                    connection.close()
    
    def add_table(self, table_name, columns_name_list, pkey_flag=True):
        create_sql = 'CREATE TABLE ' + table_name + '('
        create_sql += ', '.join(columns_name_list)
        if pkey_flag is True:
            create_sql += ', PRIMARY KEY (table_id) );'
        else:
            create_sql += ');'
        connection = None
        cursor = None
        try:
            connection = self.connect()
            cursor = connection.cursor()
            cursor.execute(create_sql)
            connection.commit()
            print('创建表', table_name, '成功')
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            print('创建表', table_name, '失败')
        finally:
            if connection:
                cursor.close()
                connection.close()
    
    def add_index(self, table_name, index_name):
        create_sql = 'CREATE INDEX index_'
        create_sql += index_name + '_' + table_name
        create_sql += ' ON '
        create_sql += table_name + '('
        create_sql += index_name + ');'
        connection = None
        cursor = None
        try:
            connection = self.connect()
            cursor = connection.cursor()
            cursor.execute(create_sql)
            connection.commit()
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            print('添加', table_name, '表索引', index_name, '失败')
        finally:
            if connection:
                cursor.close()
                connection.close()

    def add_table_with_two_index(self, table_name, columns_name_list):
        self.add_table(table_name, columns_name_list)
        self.add_index(table_name, 'start_log_id')
        self.add_index(table_name, 'end_log_id')

    def add_table_with_index(self, table_name, columns_name_list):
        self.add_table(table_name, columns_name_list)
        self.add_index(table_name, 'user_id')

    def add_table_with_index_without_pkey(self, table_name, columns_name_list):
        self.add_table(table_name, columns_name_list, False)
        self.add_index(table_name, 'user_id')
        create_sql = 'CREATE UNIQUE INDEX my_unique_index_'
        create_sql += table_name + ' ON '
        create_sql += table_name + '(user_id,es_index_name);'
        connection = None
        cursor = None
        try:
            connection = self.connect()
            cursor = connection.cursor()
            cursor.execute(create_sql)
            connection.commit()
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            print('添加', table_name, '表独立索引user_id失败')
        finally:
            if connection:
                cursor.close()
                connection.close()

    def add_item(self, table_name, key_list, one_value_list):
        sql_add = "INSERT INTO " + table_name + '('
        sql_add += ', '.join(key_list)
        sql_add += ') VALUES('
        sql_add += str(one_value_list)[1:-1]
        sql_add += ');'
        connection = None
        cursor = None
        try:
            connection = self.connect()
            cursor = connection.cursor()
            cursor.execute(sql_add)
            connection.commit()
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            print('表', table_name, 'add_item失败')
        finally:
            if connection:
                cursor.close()
                connection.close()

    def query(self, sql, error_str):
        connection = None
        cursor = None
        rs = []
        try:
            connection = self.connect()
            cursor = connection.cursor()
            cursor.execute(sql)
            rs = cursor.fetchall()
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            print(error_str)
        finally:
            if connection:
                cursor.close()
                connection.close()
        return rs

    def query_one_row(self, table_name, queried_key, queried_value):
        if not isinstance(queried_value, str):
            queried_value = str(queried_value)
        sql_select = 'SELECT * FROM ' + table_name + ' WHERE ' + queried_key + ' = \'' + queried_value + '\';'
        error_str = '高斯DB，查询表', table_name, '的一行错误'
        return self.query(sql_select, error_str)

    def query_max_row(self, table_name, queried_key):# 查询最大行
        sql_select = 'SELECT * FROM ' + table_name
        sql_select += ' WHERE ' + queried_key + ' = ( SELECT MAX('
        sql_select += queried_key + ') FROM ' + table_name + ');'
        error_str = '高斯DB，查询表', table_name, '的最大行错误'
        return self.query(sql_select, error_str)

    def query_max_id(self, table_name, queried_key):  # 查询最大行编号
        sql_select = 'SELECT MAX(' + str(queried_key) + ') FROM ' + str(table_name) + ';'
        error_str = '高斯DB，查询表', table_name, '的最大', queried_key, '错误'
        return self.query(sql_select, error_str)

    def query_multi_row_one_predict(self, table_name, queried_key, queried_value):
        if not isinstance(queried_value, str):
            queried_value = str(queried_value)
        sql_select = 'SELECT * FROM ' + table_name + ' WHERE ' + queried_key + ' >= \'' + queried_value + '\';'
        error_str = '高斯DB，查询表', table_name, '的多行（按谓词）错误'
        return self.query(sql_select, error_str)

    def query_multi_table_id(self, table_name, value_list):
        sql_select = 'SELECT * FROM ' + table_name
        sql_select += ' WHERE table_id in (' + str(value_list)[1:-1] + ');'
        error_str = '高斯DB，查询表', table_name, '的多行（table_id）错误'
        return self.query(sql_select, error_str)

    def query_window(self, table_name, queried_key1,
        queried_value1, queried_key2, queried_value2): # 查询多行,  被查询对象包含在查询窗口中
        if not isinstance(queried_value1, str):
            queried_value1 = str(queried_value1)
        if not isinstance(queried_value2, str):
            queried_value1 = str(queried_value2)
        sql_select = 'SELECT * FROM ' + table_name + ' WHERE ' + queried_key1
        sql_select += ' >= \'' + queried_value1 + '\'' + ' and ' + queried_key2 + ' <= \'' + queried_value2 + '\';'
        error_str = '高斯DB，查询表', table_name, '的多行（按window）错误'
        return self.query(sql_select, error_str)

    def query_log_id_max(self):
        sql = 'SELECT * FROM log_index_mt WHERE table_id'
        sql += ' = (SELECT max(table_id) FROM log_index_mt);'
        return self.query(sql, '高斯DB，查询表log_index_mt的log_id_max错误')

    def change_user_cnt(self, table_name, user_list, es_index_name, value_list, operate_flag):
        sql = 'INSERT INTO ' + table_name + '(user_id, es_index_name, user_cnt) VALUES(%s,%s,%s) '
        if operate_flag == 1:
            sql += ' ON DUPLICATE KEY UPDATE user_cnt = user_cnt + %s;'
        elif operate_flag == -1:
            sql += ' ON DUPLICATE KEY UPDATE user_cnt = user_cnt - %s;'
        tuple_list = []
        for i in enumerate(user_list):
            index = i[0]
            item = i[1]
            tuple_list.append((str(item), str(es_index_name), int(value_list[index]), int(value_list[index])))
            # 注意强制类型转换，排序过日志是numpy格式，要把转为python原生的格式
        connection = None
        cursor = None
        try:
            connection = self.connect()
            cursor = connection.cursor()
            cursor.executemany(sql, tuple_list)
            connection.commit()
        except (Exception, psycopg2.Error) as error:
            traceback.print_exc()
            if operate_flag == 1:
                print('表', table_name, '计数增加失败')
            elif operate_flag == -1:
                print('表', table_name, '计数减少失败')
        finally:
            if connection:
                cursor.close()
                connection.close()

    def update_user_cnt(self, table_name, user_list, es_index_name, value_list):
        self.change_user_cnt(table_name, user_list, es_index_name, value_list, 1)
        
    def substract_user_cnt(self, table_name, user_list, es_index_name, value_list):
        self.change_user_cnt(table_name, user_list, es_index_name, value_list, -1)