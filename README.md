# Automatic Speech Recognition using Wav2Vec2 & Whisper by OpenAI using Jupyter Notebook

## Introduction
Automatic Speech Recognition (ASR), or speech-to-text, is a way for computers to convert the human speech language in media files to a readable text.

## Wav2Vec2-Large-960h-Lv60 + Self-Training
### Introduction
* The large model pretrained and fine-tuned on 960 hours of Libri-Light and Librispeech on 16kHz sampled speech audio. 
* Wav2Vec2 was trained using Connectionist Temporal Classification (CTC), using Recurrent Neural Networks (RNN).
* This model is trained on speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler.
* Experiments using all labeled data of Librispeech achieve 1.9% WER on the clean test sets
* When lowering the amount of labeled data to one hour, wav2vec2.0 outperforms the previous state of the art on the 100-hour subset while using 100 times less labeled data.
* Using just ten minutes of labeled data and pre-training on 53k hours of unlabeled data still achieves 4.8/8.2 WER.

### Working
* Pass the audio vector to the wav2vec2 processor (CNN)
* Get the audio feature vector (PyTorch Tensors)
* Pass the vector into the wav2vec2 model
* Get the probabilities using logits function
* Get most likely prediction by using torch.argmax() function
* Perform inference (decode back to text & make it lowercase as it returns all the text in uppercase)

## openai/whisper-medium
### Introduction
* Trained on 680k hours of labelled data, Whisper models demonstrate a strong ability to generalize to many datasets and domains without the need for fine-tuning.
* The multilingual models were trained on both speech recognition and speech translation.
* Whisper checkpoints come in five configurations of varying model sizes. The smallest four are trained on either English-only or multilingual data.
* Gives us special tokens as well to predict the punction, sentence start/end etc.
* Since it has multilingual support, it can directly translate & transcribe into English as well.
* The Whisper model is intrinsically designed to work on audio samples of up to 30s in duration. However, by using a chunking algorithm, it can be used to transcribe audio samples of up to arbitrary length. This is possible through the Transformers Pipeline method. (Later demonstrated)
* WER of 5.9% when tested on Librispeech other dataset
* However, it results in a lower WER when presented with diverse datasets. Using 5 hours of Youtube video labelled transcriptions, it resulted in 0.15/0.23 WER

### Working
* Pass the audio to WhisperProcessor
* Extract the audio features
* Get the Special decoder tokens (start of sentence, end, punctuation etc.)
* Perform inference (decode the text)

* The Whisper architecture is a simple end-to-end approach, implemented as an encoder-decoder Transformer.
* Input audio is split into 30-second chunks, converted into a log-Mel spectrogram, and then passed into an encoder.
* A decoder is trained to predict the corresponding text caption, intermixed with special tokens that direct the single model to perform tasks such as language identification, phrase-level timestamps, multilingual speech transcription, and to-English speech translation.

## References
* "facebook/wav2vec2-large-960h-lv60-self," Hugging Face, [Online]. Available: https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self
* H. Z. A. M. M. A. Alexei Baevski, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," in 34th Conference on Neural Information Processing Systems, Vancouver, 2020.
* "Speech Recognition using Transformers in Python," PythonCode, May 2023. [Online]. Available: https://thepythoncode.com/article/speech-recognition-using-huggingface-transformers-in-python.
* "Wav2vec 2.0: Learning the structure of speech from raw audio," MetaAI, 24 September 2020. [Online]. Available: https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/
* "openai/whisper-medium," Hugging Face, [Online]. Available: https://huggingface.co/openai/whisper-medium
* J. W. K. T. X. G. B. C. M. I. S. Alec Radford, "Robust Speech Recognition via Large-Scale Weak Supervision," in Proceedings of the 40th International Conference on Machine Learning, 2023.
* "Introducing Whisper," OpenAI, 21 Spetember 2022. [Online]. Available: https://openai.com/research/whisper
* Urdu Audio from: "Text to audio," narakeet, [Online]. Available: https://www.narakeet.com/app/text-to-audio/
