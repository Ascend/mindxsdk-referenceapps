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
import hashlib
import copy
import sys


class Node(object):

    def __init__(self, node_hash = None):
        self.father = None
        self.child = [None, None]
        self.hash = node_hash

    @property
    def sibiling(self):
        if self.father is None:
            return None
        if self.father.child[0] is self:
            return self.father.child[1]
        elif self.father.child[1] is self:
            return self.father.child[0]
        return None

    @property
    def type(self):
        father = self.father
        if father is None:
            return 2 # UNKNOWN
        return 0 if (father.child[0] is self) else 1 # 0 LEFT 1 RIGHT


class MerkleTree(object):

    def __init__(self, root_node = None):
        self.root = root_node
        self.leaves = []

    def insert_list(self, new_leaf_values, already_hash):
        new_leaf_nodes = []
        for i in new_leaf_values:
            new_node_hash = i
            if already_hash is False:
                new_node_hash = hashlib.sha256(b'\x00' + i.encode()).hexdigest()
            new_node = Node(new_node_hash)
            new_leaf_nodes.append(new_node)
            self.leaves.append(new_node)
        self.update_hash_from_leaves(new_leaf_nodes) # 统一计算哈希, 从下往上  

    def update_hash_from_leaves(self, new_leaves):
        while len(new_leaves) > 1:
            if (len(new_leaves) % 2) != 0:
                empty_node = Node('#')
                new_leaves.append(empty_node)
            new_nodes = [] 
            for l, r in self._pairwise(new_leaves):
                temp_value = ''
                if l.hash != '#':
                    temp_value += l.hash 
                if r.hash != '#':
                    temp_value += r.hash 
                father_node = Node(hashlib.sha256(b'\x01' + temp_value.encode()).hexdigest())
                father_node.child = [l, r]
                l.father = father_node
                r.father = father_node
                new_nodes.append(father_node)
            new_leaves = new_nodes
        self.root = new_leaves[0]
    
    def get_proof_by_leaf_index(self, leaf_index):
        target_node = self.leaves[leaf_index]
        proofs = []
        if target_node is None:
            return None, proofs
        target = target_node
        while target != self.root:
            sibiling = target.sibiling
            if sibiling is not None:
                proofs.append(sibiling.hash + str(sibiling.type))
            else:
                proofs.append('#')
            target = target.father
        return target_node, proofs

    def _pairwise(self, iterable):
        a = iter(iterable)
        return zip(a, a)


def verify_audit_proof(node_hash, proofs, root_hash):
    target_hash = node_hash
    for i in proofs:
        temp_value = ''
        if i[:-1] == '#':
            temp_value = target_hash
        else:
            if i[-1] == '0':
                temp_value = i[:-1] + target_hash
            else:
                temp_value = target_hash + i[:-1]
        target_hash = hashlib.sha256(b'\x01' + temp_value.encode()).hexdigest()
    return target_hash == root_hash
        
