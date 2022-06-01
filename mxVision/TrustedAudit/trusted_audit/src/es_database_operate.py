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
import json
import traceback
import elasticsearch


class ESDatebase(object):
    ES_HOST_IP = '172.18.0.3'
    ES_HOST_PORT = 9200  # 9231

    def __init__(self, es_host=ES_HOST_IP, es_port=ES_HOST_PORT, timeout=3600):
        self.obj_es = elasticsearch.Elasticsearch([{'host': es_host, 'port': es_port}], timeout=3600)

    def add_index(self, index_name, schema=None):
        if self.check_index(index_name):
            print('index exists ，not create')
        else:
            resp = self.obj_es.indices.create(index=index_name, body=schema)
            print("es init resp", resp)

    def check_index(self, index_name):
        if self.obj_es.indices.exists(index_name):
            return True
        else:
            return False

    def delete_index(self, index_name):
        if self.check_index(index_name):
            resp = self.obj_es.indices.delete(index=index_name)
            print('index', index_name, 'exists ，delete', resp)
        else:
            print('index', index_name, 'not exists')

    def add_item(self, index_name, global_id, dict_item_obj, status=False):
        resp = self.obj_es.index(index=index_name, id=global_id,\
            body=dict_item_obj)
        if status:
            print("es add item resp", resp)

    def query_all(self, index_name, query_json, output_size=None, scroll_time=None):
        try:
            resp = None
            if output_size is not None:
                resp = self.obj_es.search(index=index_name,\
                    body=query_json, size=output_size)
            else:
                resp = self.obj_es.search(index=index_name,\
                    body=query_json)
            if resp['hits']['total']['value'] > 0:
                return resp['hits']['hits']
            else:
                return []
        except elasticsearch.ElasticsearchException:
            traceback.print_exc()
            return []
        finally:
            pass
            
    def get_index_of_all(self):
        stats = self.obj_es.indices.stats()
        return list(stats['indices'].keys())