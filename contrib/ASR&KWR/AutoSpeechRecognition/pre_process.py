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

import os
import io
import time
import logging
import numpy as np
import librosa
import soundfile as sf


logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s:%(asctime)s:%(message)s'
)


def read_raw_audio(audio, sample_rate=16000):
    """ Read audio data and transform it into numpy format
    Args:
      audio: it can be a filepath of audio data or bytes audio data or numpy audio data.
    Return:
      wave: numpy format audio data
    """
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate)
    elif isinstance(audio, bytes):
        wave, sample_rate_ = sf.read(io.BytesIO(audio))
        wave = np.asfortranarray(wave)
        if sample_rate_ != sample_rate:
            wave = librosa.resample(wave, sample_rate_, sample_rate)
    elif isinstance(audio, np.ndarray):
        return audio
    else:
        raise ValueError("input audio must be either a path or bytes")
    return wave


def normalize_audio_feature(audio_feature: np.ndarray, per_feature=False):
    """ Mean and variance normalization """
    axis = 0 if per_feature else None
    mean = np.mean(audio_feature, axis=axis)
    std_dev = np.std(audio_feature, axis=axis) + 1e-9
    normalized = (audio_feature - mean) / std_dev
    return normalized


def normalize_signal(signal: np.ndarray):
    """ Normailize signal to [-1, 1] range """
    gain = 1.0 / (np.max(np.abs(signal)) + 1e-9)
    return signal * gain


def preemphasis(signal: np.ndarray, coeff=0.97):
    if not coeff or coeff <= 0.0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def deemphasis(signal: np.ndarray, coeff=0.97):
    if not coeff or coeff <= 0.0:
        return signal
    de_emphasis_signal = np.zeros(signal.shape[0], dtype=np.float32)
    de_emphasis_signal[0] = signal[0]
    for i in range(1, signal.shape[0], 1):
        de_emphasis_signal[i] = coeff * de_emphasis_signal[i - 1] + signal[i]
    return de_emphasis_signal


class SpeechFeaturizer(object):
    '''SpeechFeaturizer offers 3 types of feature extraction: \
        'mfcc', 'logfbank' , 'spectrogram'.
    '''

    def __init__(
            self,
            sample_rate=16000,
            frame_ms=25,
            stride_ms=10,
            num_feature_bins=80,
            feature_type='logfbank',
            delta=False,
            delta_delta=False,
            pitch=False,
            preemphasis_rate=0.97,
            is_normalize_signal=True,
            is_normalize_feature=True,
            is_normalize_per_feature=False):

        # Samples
        self.sample_rate = sample_rate
        self.frame_length = int(self.sample_rate * (frame_ms / 1000))
        self.frame_step = int(self.sample_rate * (stride_ms / 1000))
        # Features
        self.num_feature_bins = num_feature_bins
        self.feature_type = feature_type
        self.delta = delta
        self.delta_delta = delta_delta
        self.pitch = pitch
        self.preemphasis_rate = preemphasis_rate
        # Normalization
        self.is_normalize_signal = is_normalize_signal
        self.is_normalize_feature = is_normalize_feature
        self.is_normalize_per_feature = is_normalize_per_feature

    def load_wav(self, path):
        wav = read_raw_audio(path, self.sample_rate)
        return wav

    def compute_time_dim(self, seconds: float) -> int:
        # implementation using pad "reflect" with n_fft // 2
        total_frames = seconds * self.sample_rate + 2 * (self.frame_length //
                                                         2)
        return int(1 + (total_frames - self.frame_length) // self.frame_step)

    def pad_signal(self, wavs, max_length=None):
        '''padding data before featrure extraction'''
        if not max_length:
            max_length = self.sample_rate*10
        wavs = np.pad(wavs, (0, max_length-wavs.shape[0]), 'constant')
        return wavs

    @classmethod
    def pad_feat(cls, feat, max_length=1001):
        '''padding data after feature extraction'''
        if feat.shape[0] > max_length:
            feat = feat[0:max_length+1]
        else:
            feat = np.pad(
                feat, ((0, max_length-feat.shape[0]),
                       (0, 0), (0, 0)), 'constant')
        return feat

    def compute_feature_dim(self) -> tuple:
        channel_dim = 1

        if self.delta:
            channel_dim += 1

        if self.delta_delta:
            channel_dim += 1

        if self.pitch:
            channel_dim += 1

        return self.num_feature_bins, channel_dim

    def extract(self, signal: np.ndarray) -> np.ndarray:
        if self.is_normalize_signal:
            signal = normalize_signal(signal)
        signal = preemphasis(signal, self.preemphasis_rate)

        if self.feature_type == "mfcc":
            features = self._compute_mfcc_feature(signal)
        elif self.feature_type == "logfbank":
            features = self._compute_logfbank_feature(signal)
        elif self.feature_type == "spectrogram":
            features = self._compute_spectrogram_feature(signal)
        else:
            raise ValueError(
                "feature_type must be either 'mfcc', 'logfbank' or \
                'spectrogram'"
            )

        original_features = np.copy(features)

        if self.is_normalize_feature:
            features = normalize_audio_feature(
                features, per_feature=self.is_normalize_per_feature)

        features = np.expand_dims(features, axis=-1)

        if self.delta:
            delta = librosa.feature.delta(original_features.T).T
            if self.is_normalize_feature:
                delta = normalize_audio_feature(
                    delta, per_feature=self.is_normalize_per_feature)
            features = np.concatenate(
                [features, np.expand_dims(delta, axis=-1)], axis=-1)

        if self.delta_delta:
            delta_delta = librosa.feature.delta(original_features.T, order=2).T
            if self.is_normalize_feature:
                delta_delta = normalize_audio_feature(
                    delta_delta, per_feature=self.is_normalize_per_feature)
            features = np.concatenate(
                [features, np.expand_dims(delta_delta, axis=-1)], axis=-1)

        if self.pitch:
            pitches = self._compute_pitch_feature(signal)
            if self.is_normalize_feature:
                pitches = normalize_audio_feature(
                    pitches, per_feature=self.is_normalize_per_feature)
            features = np.concatenate(
                [features, np.expand_dims(pitches, axis=-1)], axis=-1)

        return features

    def _compute_pitch_feature(self, signal: np.ndarray) -> np.ndarray:
        pitches, _ = librosa.core.piptrack(y=signal,
                                           sr=self.sample_rate,
                                           n_fft=self.frame_length,
                                           hop_length=self.frame_step,
                                           fmin=0,
                                           fmax=int(self.sample_rate / 2),
                                           win_length=self.frame_length,
                                           center=True)

        pitches = pitches.T

        # num_features for spectrogram should be <= (sample_rate * window_size // 2 + 1)
        if (self.num_feature_bins > self.frame_length // 2 + 1):
            logging.warning(
                "num_features for spectrogram should be <= (sample_rate * window_size // 2 + 1)")

        return pitches[:, :self.num_feature_bins]

    def _compute_spectrogram_feature(self, signal: np.ndarray) -> np.ndarray:
        powspec = np.abs(
            librosa.core.stft(signal,
                              n_fft=self.frame_length,
                              hop_length=self.frame_step,
                              win_length=self.frame_length,
                              center=True))

        # remove small bins
        features = 20 * np.log10(powspec.T)

        # num_features for spectrogram should be <= (sample_rate * window_size // 2 + 1)
        if (self.num_feature_bins > self.frame_length // 2 + 1):
            logging.warning(
                "num_features for spectrogram should be <= (sample_rate * window_size // 2 + 1)")

        # cut high frequency part, keep num_feature_bins features
        features = features[:, :self.num_feature_bins]

        return features

    def _compute_mfcc_feature(self, signal: np.ndarray) -> np.ndarray:
        log_power_Mel_spectrogram = np.square(
            np.abs(
                librosa.core.stft(signal,
                                  n_fft=self.frame_length,
                                  hop_length=self.frame_step,
                                  win_length=self.frame_length,
                                  center=True)))

        mel_basis = librosa.filters.mel(self.sample_rate,
                                        self.frame_length,
                                        n_mels=128,
                                        fmin=0,
                                        fmax=int(self.sample_rate / 2))

        mfcc = librosa.feature.mfcc(
            sr=self.sample_rate,
            S=librosa.core.power_to_db(
                np.dot(mel_basis, log_power_Mel_spectrogram) + 1e-20),
            n_mfcc=self.num_feature_bins)

        return mfcc.T

    def _compute_logfbank_feature(self, signal: np.ndarray) -> np.ndarray:
        log_power_Mel_spectrogram = np.square(
            np.abs(
                librosa.core.stft(signal,
                                  n_fft=self.frame_length,
                                  hop_length=self.frame_step,
                                  win_length=self.frame_length,
                                  center=True)))

        mel_basis = librosa.filters.mel(self.sample_rate,
                                        self.frame_length,
                                        n_mels=self.num_feature_bins,
                                        fmin=0,
                                        fmax=int(self.sample_rate / 2))

        return np.log(np.dot(mel_basis, log_power_Mel_spectrogram) + 1e-20).T


def make_one_data(wav_path, speech_feat: SpeechFeaturizer):
    '''Extract the features of a speech'''

    wav_data = speech_feat.load_wav(wav_path)
    feat_data = speech_feat.extract(wav_data)
    # Padding the data after feature extraction
    feat_data = speech_feat.pad_feat(feat_data)
    length = feat_data.shape[0]//4
    return feat_data, length


def make_model_input(wav_path_list):
    '''Build batch data with batch sizes of 1, 2, 4, 8.

    Args:
      wav_path_list: The list of wav file paths with the length of 1, 2, 4, 8.

    Returns:
      wav_data_batch: one batch data of extracted speech featrues.
      length_data_batch: one batch data of the length which shows how many \
    words one speech contains.
    '''
    wav_data_list = []
    length_data_list = []
    speech_feat = SpeechFeaturizer()
    for wav_path in wav_path_list:
        wav_data, length_data = make_one_data(wav_path, speech_feat)
        wav_data_list.append(wav_data)
        length_data_list.append([length_data])
    wav_data_batch = np.array(wav_data_list, dtype='float32')
    length_data_batch = np.array(length_data_list, dtype='int32')
    return wav_data_batch, length_data_batch


if __name__ == '__main__':
    # Test the time of feature extraction
    start = time.time()
    wav_path = r"./BAC009S0764W0121.wav"
    wav_path_list = [wav_path]
    wav_data_batch, length_data_batch = make_model_input(wav_path_list)
    end = time.time()
    print("total time: ", end - start)
