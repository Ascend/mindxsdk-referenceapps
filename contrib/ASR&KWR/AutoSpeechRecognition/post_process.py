# Copyright 2021 Huawei Technologies Co., Ltd
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

import codecs


class TextFeaturizer:
    """
    Establishes a mapping of indexs to tokens.
    0 unused, 1 for start 'S' and 2 for end '\S'.
    """
    def __init__(self, token_path):

        self.num_classes = 0
        self.vocab_array = []
        self.token_to_index = {}
        self.index_to_token = {}

        lines = []
        with codecs.open(token_path, "r", "utf-8") as fin:
            lines.extend(fin.readlines())

        index = 1
        for line in lines:
            # Strip the '\n' char
            line = line.strip()
            # Skip comment line, empty line
            if line.startswith("#") or not line or line == "\n":
                continue
            self.token_to_index[line] = index
            self.index_to_token[index] = line
            self.vocab_array.append(line)
            index += 1

        self.num_classes = len(self.vocab_array)

    def startid(self):
        return self.token_to_index['S']

    def endid(self):
        return self.token_to_index['/S']

    def encode(self, tokens):
        """
        Convert string to a list of integers
        Args:
            text: string (sequence of characters)

        Returns:
            sequence of ints
        """
        feats = [self.token_to_index[token] for token in tokens]
        return feats

    def decode(self, ids):
        """Convert a list of integers to a list of tokens"""

        tokens = [self.index_to_token[id] for id in ids]
        return tokens

    def deocde_without_start_end(self, ids):
        """Convert a list of integers to a list of tokens \
        without 'S' and '\S' """
        tokens = []
        for i in ids:
            if i == self.startid():
                continue
            elif i == self.endid():
                break
            else:
                tokens.append(self.index_to_token[i])
        return tokens


if __name__ == '__main__':
    # Sample test
    # token_path is the filepath of the dictionary you want to decode.
    token_path = "LMmodel/lm_tokens.txt"
    text_feat = TextFeaturizer(token_path)

    sample_string = '今天是个好日子'
    sample_input = text_feat.encode(sample_string)

    sample_result = text_feat.decode(sample_input)
    print("sample result: ", sample_result)
