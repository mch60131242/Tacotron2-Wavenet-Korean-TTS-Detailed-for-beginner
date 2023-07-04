# Multi-Speaker Tocotron2 + Wavenet Vocoder + Korean TTS
Tacotron2 모델과 Wavenet Vocoder를 결합하여  한국어 TTS구현하는 project입니다.
Tacotron2 모델을 Multi-Speaker모델로 확장했습니다.
Tacotron2 모델을 통해 기본적으로 음성을 생성하는 것이 가능합니다. 이후에 Vocoder를 통해 음성을 더욱 자연스럽게 생성이 가능합니다.



Based on 
- https://github.com/keithito/tacotron
- https://github.com/carpedm20/multi-speaker-tacotron-tensorflow
- https://github.com/Rayhane-mamah/Tacotron-2
- https://github.com/hccho2/Tacotron-Wavenet-Vocoder
- https://github.com/MoorDev/Tacotron2-Wavenet-Korean-TTS


## Tacotron 2
- Tacotron 모델에 관한 설명은 이전 [repo](https://github.com/hccho2/Tacotron-Wavenet-Vocoder) 참고하시면 됩니다.
- [Tacotron2](https://arxiv.org/abs/1712.05884)에서는 모델 구조도 바뀌었고, Location Sensitive Attention, Stop Token, Vocoder로 Wavenet을 제안하고 있다.
- Tacotron2의 대표적인 구현은 [Rayhane-mamah](https://github.com/Rayhane-mamah/Tacotron-2)입니다. 이 역시, [keithito](https://github.com/keithito/tacotron), [r9y9](https://github.com/r9y9/wavenet_vocoder)의 코드를 기반으로 발전된 것이다.

## This Project
* Tacotron2 모델로 한국어 TTS를 만드는 것이 목표입니다.
* [Rayhane-mamah](https://github.com/Rayhane-mamah/Tacotron-2)의 구현은 Customization된 Layer를 많이 사용했는데, 제가 보기에는 너무 복잡하게 한 것 같아, Cumomization Layer를 많이 줄이고, Tensorflow에 구현되어 있는 Layer를 많이 활용했습니다.
* teacher forcing 방식의 train sample은 2000 step부터, free forcing 방식의 test sample은 3000 step부터 알아들을 수 있는 정도의 음성을 만들기 시작합니다.
## 단계별 실행

### 실행 순서
- Data 생성: 한국어 data의 생성은 이전 [repo](https://github.com/hccho2/Tacotron-Wavenet-Vocoder) 참고하시면 됩니다.- 이부분을 통해 데이터를 어떻게 구성하고
Json파일을 어떻게 구성해야하는지 확인하면 됩니다.
- 생성된 Data는 아래의 'data_paths'에 지정하면 된다.
- -Data의 형식은 json 파일에 "file_path : "텍스트내용"의 형태로 만들면된다.
- tacotron training 후, synthesize.py로 test하면 됩니다..
- wavenet training 후, generate.py로 test(tacotron이 만들지 않은 mel spectrogram으로 test할 수도 있고, tacotron이 만든 mel spectrogram을 사용할 수도 있다.)
- 2개 모델 모두 train 후, tacotron에서 생성한 mel spectrogram을 wavent에 local condition으로 넣어 test하면 된다.


### Tacotron2 Training
- train_tacotron2.py 내에서 '--data_paths'를 지정한 후, train할 수 있다. data_path는 여러개의 데이터 디렉토리를 지정할 수 있습니다.
```
parser.add_argument('--data_paths', default='.\\data\\moon,.\\data\\son')
Data_paths를 .py파일을 수정하거나 터미널에서 실행시에 직접 전달하기를 통해 설정해주면 됩니다.
```
- train을 이어서 계속하는 경우에는 '--load_path'를 지정해 주면 된다.
```
parser.add_argument('--load_path', default='logdir-tacotron2/moon+son_2019-02-27_00-21-42')
-train.py에 기본적으로 None아니면 작동되지않기떄문에 들어가서 처음 실행시 default=None으로 지정해주어야 한다.
-기본적으로는 None으로 지정되어 있으며 이것은 load 하지않고 처음부터 "다시"학습을 시작하는 것을 의미합니다.
```

- model_type은 'single' 또는 ' multi-speaker'로 지정할 수 있다. speaker가 1명 일 때는, hparams.py의 model_type = 'single'로 하고 train_tacotron2.py 내에서 '--data_paths'를 1개만 넣어주면 된다.
```
parser.add_argument('--data_paths', default='D:\\Tacotron2\\data\\moon')
```
- 하이퍼파라메터를 hparmas.py에서 argument를 train_tacotron2.py에서 다 설정했기 때문에, train 실행은 다음과 같이 단순합니다.
> python train_tacotron2.py
- train 후, 음성을 생성하려면 다음과 같이 하면 된다. '--num_speaker', '--speaker_id'는 잘 지정되어야 한다.
- 만약 Single speak인 경우에는 --num_speaker를 1로 --speaker_id를 0으로 지정해주면 된다.
> python synthesizer.py --load_path logdir-tacotron2/moon+son_2019-02-27_00-21-42 --num_speakers 2 --speaker_id 0 --text "내가 원하는 내용."
- 이 부분에서 logdir에서 자신이 원하는 스텝의 모델을 설정하여 작동 할 수 있다.
- synthesizer.py를 할경우 log_dir 에 generate 폴더에 wav파일과 npy파일이 생성된다.


### Wavenet Vocoder Training
- Wavenet의 트레이닝을 진행하지 않는다 하더라도 tacotron2에 지정된 기본 알고리즘에 의해 음성을 생성 할 수는 있으나 Wavenet으로 만들어낸 음성이 더 깔끔하다.
- 
- train_vocoder.py 내에서 '--data_dir'를 지정한 후, train할 수 있다.
- memory 부족으로 training 되지 않거나 너무 느리면, hyper paramerter 중 sample_size를 줄이면 된다. 물론 batch_size를 줄일 수도 있다.
```
DATA_DIRECTORY =  'D:\\Tacotron2\\data\\moon,D:\\Tacotron2\\data\\son'
parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY, help='The directory containing data')
```
- train을 이어서 계속하는 경우에는 '--logdir'를 지정해 주면 된다.
```
LOGDIR = './/logdir-wavenet//train//2018-12-21T22-58-10'
parser.add_argument('--logdir', type=str, default=LOGDIR)
-train.py에 기본적으로 None아니면 작동되지않기떄문에 들어가서 처음 실행시 default=None으로 지정해주어야 한다.
- wavenet train 후, tacotron이 생성한 mel spectrogram(npy파일)을 local condition으로 넣어서 TTS의 최종 결과를 얻을 수 있다.
> python generate.py --mel ./logdir-wavenet/mel-moon.npy --gc_cardinality 2 --gc_id 0 ./logdir-wavenet/train/2018-12-21T22-58-10

### Result
- Tacotron의 batch_size = 32, Wavenet의 batch_size=8. 
- Tacotron은 step 50K, Wavenet은 60K 만큼 train.



# Requirement
 - Ubuntu 18.04
 - Tensorflow 1.14.0
 - Python 3.6
 - ffmpeg
 - llvm 10
 - librosa
 - requests
 > pip install -r requre.txt

 - Installed Package
 -- absl-py (0.11.0)
 --appdirs (1.4.4)
 --asn1crypto (0.24.0)
--astor (0.8.1)
--attrs (17.4.0)
--audioread (2.1.9)
--Automat (0.6.0)
--blinker (1.4)
--cached-property (1.5.2)
--certifi (2018.1.18)
--cffi (1.14.4)
--chardet (3.0.4)
--click (6.7)
--cloud-init (20.4)
--colorama (0.4.4)
--command-not-found (0.3)
--configobj (5.0.6)
--constantly (15.1.0)
--cryptography (2.1.4)
--cycler (0.10.0)
--decorator (4.4.2)
--gast (0.2.2)
--google-pasta (0.2.0)
--grpcio (1.35.0)
--h5py (3.1.0)
--httplib2 (0.9.2)
--hyperlink (17.3.1)
--idna (2.6)
--importlib-metadata (3.4.0)
--incremental (16.10.1)
--inflect (5.0.2)
--jamo (0.4.1)
--Jinja2 (2.10)
--joblib (1.0.0)
--jsonpatch (1.16)
--jsonpointer (1.10)
--jsonschema (2.6.0)
--Keras-Applications (1.0.8)
--Keras-Preprocessing (1.1.2)
--keyring (10.6.0)
--keyrings.alt (3.0)
--kiwisolver (1.3.1)
--language-selector (0.1)
--librosa (0.8.0)
--llvmlite (0.35.0)
--Markdown (3.3.3)
--MarkupSafe (1.0)
--matplotlib (3.3.3)
--mock (4.0.3)
--netifaces (0.10.4)
--nltk (3.5)
--numba (0.52.0)
--numpy (1.19.5)
--oauthlib (2.0.6)
--opt-einsum (3.3.0)
--packaging (20.8)
--PAM (0.4.2)
--Pillow (8.1.0)
--pooch (1.3.0)
--protobuf (3.14.0)
--pyasn1 (0.4.2)
--pyasn1-modules (0.2.1)
--pycparser (2.20)
--pycrypto (2.6.1)
--Pygments (2.2.0)
--pygobject (3.26.1)
--PyJWT (1.5.3)
--pyOpenSSL (17.5.0)
--pyparsing (2.4.7)
--pyserial (3.4)
--python-apt (1.6.5+ubuntu0.5)
--python-dateutil (2.8.1)
--pyxdg (0.25)
--PyYAML (3.12)
--regex (2020.11.13)
--requests (2.18.4)
--requests-unixsocket (0.1.5)
--resampy (0.2.2)
--scikit-learn (0.24.1)
--scipy (1.5.4)
--SecretStorage (2.3.1)
--service-identity (16.0.0)
--six (1.15.0)
--SoundFile (0.10.3.post1)
--ssh-import-id (5.7)
--systemd-python (234)
--termcolor (1.1.0)
--threadpoolctl (2.1.0)
--tqdm (4.56.0)
--Twisted (17.9.0)
--typing-extensions (3.7.4.3)
--unattended-upgrades (0.1)
--Unidecode (1.1.2)
--urllib3 (1.22)
--Werkzeug (1.0.1)
--wheel (0.36.2)
--wrapt (1.12.1)
--zipp (3.4.0)
--zope.interface (4.3.2)

## 환경 선택 ##
"""데이터를 학습함에 있어서 리눅스와 윈도우 에서의 실행 방식이 다르니 자신의 환경이 어떠한 환경인지를 생각하고 학습을 진행한다"""

## 데이터 탐색 ##
"""다행히 Ai-Hub나 유투브를 통해서 데이터를 구하거나 모델을 탐색할 수 있는 기회가 많이 있으므로 어느정도 시간을 가지고 탐색을 진행할 것을 권고한다.
또한 데이터의 종류나 화자의 발음, 노이즈 상태, 억양 에 따라서 데이터의 학습 시간이 달라지기 때문에 현재 자신이 가지고 있는 기기의 성능이나 학습 시간 등을 고려해서
데이터를 선택하는 것이 좋다."""


## 데이터 전처리##
"""Data path 와 그 에 따른 데이터의 텍스트를 한 Json 파일에 정리해야 함으로, 그것이 틀린다면 학습이 진행되지 않거나 진행 되더라도 올바른 학습이 진행되기 어려움
학습에 사용될 데이터의 용량이 충분해야 하며, 노이즈가 섞여 있는 데이터는 학습률이 내려가므로 데이터를 수집하거나 생성시에 주의해야한다.
"""

## 학습시의 고려사항##
"""모델을 상황에 따라 수정하는 연습을 할 필요가 있다. 따라서 학습을 진행하기 이전에 스스로 어느정도 모델을 학습할 시간을 가지는 것도 중요하다고 생각한다.
데이터의 학습에 오랜 시간이 필요하기 때문에, 이 모델을 통해 공부 또는 구현해 볼 생각이라면, 높은 성능의 GPU를 사용하고 이를 Tensorflow에 연결하는 방법도 이해하고 있어야 한다.(이 모델 개발 시에 사용한 Tensorflow는2.0 이전 버젼이기 때문에 Gpu를 사용하기 위해서는 다른 방법이 필요하다.)
Tacotron2를 이용해서 만든 데이터도 생성을 해서 들어오면 WaveNet(Vocoder)를 사용하기 이전임에도 좋은 성능을 보여준다.
Vocoder학습에는 Tacotron2에 사용 되는 시간보다 2배 이상의 시간이 들어가기 때문에 적절한 성능의 모델을 만들기 위해서는 꼭 시간을 확보해야한다.
학습 중간 중간 학습의 상태를 확인하고 generate, synthesize 해보는 습관을 가지면 좋다."""

