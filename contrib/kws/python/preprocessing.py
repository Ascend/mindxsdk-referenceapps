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
import numpy as np
import os
import librosa
from sklearn import preprocessing
from overrides import overrides


class AudioTools(object):
    @classmethod
    def vad(cls, wav_data, sample_rate, frame_length):
        """
        The audio is roughly removed from both ends of the mute, single threshold short time average amplitude
        Args:
            wav_data:  audio data
            sample_rate: sample rate(int)
            frame_length: (frame length/ms)
        Return:
            audio data after thr vad(numpy)
        """
        y = wav_data / np.max(np.abs(wav_data))
        audio_length = len(wav_data)
        frame_length = frame_length * sample_rate // 1000
        frame_nums = audio_length // frame_length
        frame_energy = []
        for i in range(frame_nums):
            frame = y[i * frame_length:(i + 1) * frame_length]
            # the short-time average amplitude
            energy = np.sum(abs(frame))
            frame_energy.append(energy)
        max_energy = max(frame_energy)
        min_energy = min(frame_energy)
        silence_threshold = min(frame_energy) + (1 / 48) * (max_energy - min_energy)
        judge_bool = frame_energy > silence_threshold
        voice_index = np.where(judge_bool == True)[0]
        # the subscript of the first frame is greater than the mute threshold, and three frames are reserved to mute
        voice_start_index = max(0, voice_index[0] - 3)
        voice_end_index = min(voice_index[-1] + 3, len(frame_energy) - 1)
        return wav_data[voice_start_index * frame_length: (voice_end_index + 1) * frame_length]

    @classmethod
    def feature_padded(cls, feature, feat_dim, max_len=1000, padded_type="zero"):
        """
        Fill/truncate features, truncate features when max_len is exceeded,
        and fill features when max_len is less than 1000(frames) by default, which is about 10s
        Args:
            feature: (feat_dim, time_steps)
            feat_dim: dimension of the feature
            max_len: max frame length
            padded_type:
                         "zero": Fill in 0 to complete the task.
                         "copy": Copy the previous content to complete the task. It is applicable to speaker tasks
        Return:
            padded_feat: The feature after filled or truncated
            feat_real_len: If filled: the time step without feature filling
                           If truncated: the time step after truncation
        """
        feature_len = feature.shape[1]
        if max_len <= feature_len:
            padded_feat = feature[:, :max_len]
            feat_real_len = max_len
        else:
            feat_real_len = feature_len
            padded_feat = cls._pad_feature(feature, feat_dim, max_len, padded_type)
        return padded_feat, feat_real_len
    
    @staticmethod
    def _pad_feature(feature, feat_dim, max_len=1000, padded_type="zero"):
        """
        Fill/truncate features, truncate features when max_len is exceeded,
        and fill features when max_len is less than 1000(frames) by default, which is about 10s
        Args:
            feature: (feat_dim, time_steps)
            feat_dim: dimension of the feature
            max_len: max frame length
            padded_type:
                         "zero": Fill in 0 to complete the task.
                         "copy": Copy the previous content to complete the task. It is applicable to speaker tasks
        Return:
            padded_feat: The feature after filled or truncated
        """
        feature_len = feature.shape[1]
        if padded_type == "zero":
            padded_feat = np.zeros((feat_dim, max_len))
            padded_feat[:, :feature_len] = feature
        elif padded_type == "copy":
            padded_feat = np.zeros((feat_dim, max_len))
            n = max_len // feature_len + 1
            # Copy it n times along the time axis
            feature = np.tile(feature, (1, n))
            # truncate
            padded_feat[:, :] = feature[:, :max_len]
        else:
            raise TypeError("padded_type must be 'zero' or 'copy', but received {}".format(padded_type))
        return padded_feat


class BaseExtract(AudioTools):
    def __init__(self, frame_length=25, frame_shift=10, max_len=1000,
                 sr=16000, padded_type="zero", mean_std_path=None):
        self._sr = sr
        self._frame_length = frame_length
        self._frame_shift = frame_shift
        self._max_len = max_len
        self._padded_type = padded_type
        self._mean_std_path = mean_std_path

    def extract_feature(self, wav_path, feat_dim, scale_flag=False):
        try:
            wav, sr = self._read_wav(wav_path, self.sr)
        except:
            raise ValueError("Read file {} filed".format(wav_path))
        wav = self.vad(wav_data=wav, sample_rate=sr, frame_length=20)
        wav = self._normalize(wav)
        wav_feature = self._feature_extract(wav, sr, feat_dim)
        if scale_flag:
            wav_feature = self._standardize(wav_feature, self._mean_std_path)
        wav_feature, feat_real_len = self.feature_padded(wav_feature, feat_dim, self._max_len, self._padded_type)
        wav_feature = np.array(wav_feature, dtype=np.float32)
        return wav_feature, feat_real_len

    def _read_wav(self, wav_path, sr):
        '''
        read wav
        '''
        return librosa.load(wav_path, sr)

    def _normalize(self, x):
        '''
        normalize
        '''
        y = x.astype(np.float32)
        normalization_factor = 1 / (np.max(np.abs(y))+1e-5)
        y = y * normalization_factor
        return y
    
    def _standardize(self, x, mean_std_path=None):
        try:
            data = np.load(mean_std_path)
            mean = data["mean"]
            std = data["std"]
            x = (x.T - mean) / (std + 1e-6)
            x = x.T
        except:
            x = preprocessing.scale(x, axis=1)
        return x

    def _feature_extract(self, wav, sr, feat_dim):
        raise NotImplementedError

    @property
    def sr(self):
        return self._sr

    @property
    def frame_length(self):
        return self._frame_length

    @property
    def frame_shift(self):
        return self._frame_shift

    @property
    def max_len(self):
        return self._max_len

    @property
    def padded_type(self):
        return self._padded_type

    @property
    def mean_std_path(self):
        return self._mean_std_path

    @sr.setter
    def sr(self, value):
        if isinstance(value, int):
            self._sr = value
        else:
            raise TypeError("sr type must be int but received {}".format(type(value)))


class ExtractMfcc(BaseExtract):
    @overrides
    def _feature_extract(self, wav, sr, feat_dim):
        mel_spectrogram = librosa.feature.melspectrogram(wav,
                                                         sr=sr,
                                                         n_mels=feat_dim,
                                                         hop_length=(sr * self.frame_shift) // 1000,
                                                         n_fft=(sr * self.frame_length) // 1000,
                                                         fmin=20,
                                                         fmax=sr // 2)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectrogram), n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
        mfcc_comb = np.concatenate([mfcc, mfcc_delta, mfcc_delta_delta], axis=0)
        return mfcc_comb


class ExtractLogmel(BaseExtract):
    @overrides
    def _feature_extract(self, wav, sr, feat_dim):
        mel_spectrogram = librosa.feature.melspectrogram(wav,
                                                         sr=sr,
                                                         n_mels=feat_dim,
                                                         hop_length=(sr * self.frame_shift) // 1000,
                                                         n_fft=(sr * self.frame_length) // 1000,
                                                         fmin=20,
                                                         fmax=sr // 2)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram
