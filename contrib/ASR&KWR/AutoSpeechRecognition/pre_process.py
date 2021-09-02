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
        # If the sampling rates do not match, resampling will be done.
        if sample_rate_ != sample_rate:
            wave = librosa.resample(wave, sample_rate_, sample_rate)
    elif isinstance(audio, np.ndarray):
        return audio
    else:
        raise ValueError("input audio must be either a path or bytes")
    return wave


def normalize_audio_feature(audio_feature: np.ndarray, per_feature=False):
    """ mean and variance normalization """
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
    '''improve the high frequency data of speech'''
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
        # use 0 for padding
        wavs = np.pad(wavs, (0, max_length-wavs.shape[0]), 'constant')
        return wavs

    # Defining it as a class method can make your program run faster
    @classmethod
    def pad_feat(cls, feat, max_length=1001):
        '''padding data after feature extraction'''

        # Truncate data that exceeds the maximum length
        if feat.shape[0] > max_length:
            feat = feat[0:max_length]
        else:
            # Use 0 for padding to max_length at axis 0.
            feat = np.pad(
                feat, ((0, max_length-feat.shape[0]),
                       (0, 0), (0, 0)), 'constant')
        return feat

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """feature extraction according to feature type"""
        # 1. Normalize signal
        if self.is_normalize_signal:
            signal = normalize_signal(signal)
        signal = preemphasis(signal, self.preemphasis_rate)
        # 2. Compute feature
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
        # 3. Normalize feature
        if self.is_normalize_feature:
            # mean and variance normalization
            features = normalize_audio_feature(
                features, per_feature=self.is_normalize_per_feature)

        features = np.expand_dims(features, axis=-1)
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
        if self.num_feature_bins > self.frame_length // 2 + 1:
            logging.warning(
                "num_features for spectrogram should be <= (sample_rate * window_size // 2 + 1)")

        return pitches[:, :self.num_feature_bins]

    # compute spectrogram feature
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
        if self.num_feature_bins > self.frame_length // 2 + 1:
            logging.warning(
                "num_features for spectrogram should be <= (sample_rate * window_size // 2 + 1)")

        # cut high frequency part, keep num_feature_bins features
        features = features[:, :self.num_feature_bins]

        return features

    # compute mfcc feature
    def _compute_mfcc_feature(self, signal: np.ndarray) -> np.ndarray:
        # 1. The Fourier transform is used to get the mel spectral characteristics
        log_power_mel_spectrogram = np.square(
            np.abs(
                librosa.core.stft(signal,
                                  n_fft=self.frame_length,
                                  hop_length=self.frame_step,
                                  win_length=self.frame_length,
                                  center=True)))
        # 2. Use librosa tool to get mel basis
        mel_basis = librosa.filters.mel(self.sample_rate,
                                        self.frame_length,
                                        n_mels=128,
                                        fmin=0,
                                        fmax=int(self.sample_rate / 2))
        # 3. Obtain mfcc characteristics
        mfcc = librosa.feature.mfcc(
            sr=self.sample_rate,
            S=librosa.core.power_to_db(
                np.dot(mel_basis, log_power_mel_spectrogram) + 1e-20),
            n_mfcc=self.num_feature_bins)

        return mfcc.T

    # compute logbank feature
    def _compute_logfbank_feature(self, signal: np.ndarray) -> np.ndarray:
        # 1. The Fourier transform is used to get the mel spectral characteristics
        log_power_mel_spectrogram = np.square(
            np.abs(
                librosa.core.stft(signal,
                                  n_fft=self.frame_length,
                                  hop_length=self.frame_step,
                                  win_length=self.frame_length,
                                  center=True)))

        # 2. Use librosa tool to get mel basis
        mel_basis = librosa.filters.mel(self.sample_rate,
                                        self.frame_length,
                                        n_mels=self.num_feature_bins,
                                        fmin=0,
                                        fmax=int(self.sample_rate / 2))

        # 3. Dot product mel_basi and log_power_mel_spectrogram to get logfbank feature
        return np.log(np.dot(mel_basis, log_power_mel_spectrogram) + 1e-20).T


def make_one_data(wav_path, speech_feat: SpeechFeaturizer):
    '''Extract the features of a speech'''

    # Load data and extract features.
    wav_data = speech_feat.load_wav(wav_path)
    feat_data = speech_feat.extract(wav_data)
    # Padding the data after feature extraction.
    feat_data = speech_feat.pad_feat(feat_data)
    # Calculates the length of the text corresponding to the speech data.
    length = feat_data.shape[0]//4
    return feat_data, length


def make_model_input(wav_path_list):
    '''Build batch data with batch sizes of length of wav path list.

    Args:
      wav_path_list: The list of wav file paths.

    Returns:
      wav_data_batch: one batch data of extracted speech featrues.
      length_data_batch: one batch data of the length which shows how many \
    words one speech contains.
    '''
    wav_data_list = []
    length_data_list = []
    # Instantiate SpeechFeaturizer
    speech_feat = SpeechFeaturizer()
    for wav_path in wav_path_list:
        wav_data, length_data = make_one_data(wav_path, speech_feat)
        wav_data_list.append(wav_data)
        length_data_list.append([length_data])
    # Convert list to NumPy data
    wav_data_batch = np.array(wav_data_list, dtype='float32')
    length_data_batch = np.array(length_data_list, dtype='int32')
    return wav_data_batch, length_data_batch


if __name__ == '__main__':
    # Test the time of feature extraction
    start = time.time()
    wav_path = r"./data/BAC009S0008W0121.wav"
    # Put the path to the recognized voice file in a list
    wav_path_list = [wav_path]
    feat_data_batch, length_data_batch = make_model_input(wav_path_list)
    end = time.time()
    # print the time of feature extraction
    print("total time: ", end - start)
