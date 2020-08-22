# KominGPT

[![Code Coverage](https://codecov.io/gh/HephaestusProject/template/branch/master/graph/badge.svg)](https://codecov.io/gh/HephaestusProject/template)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract

This is Korean min GPT and here's original GPT's abstract below.

Natural language understanding comprises a wide range of diverse tasks such
as textual entailment, question answering, semantic similarity assessment, and
document classification. Although large unlabeled text corpora are abundant,
labeled data for learning these specific tasks is scarce, making it challenging for
discriminatively trained models to perform adequately. We demonstrate that large
gains on these tasks can be realized by generative pre-training of a language model
on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each
specific task. In contrast to previous approaches, we make use of task-aware input
transformations during fine-tuning to achieve effective transfer while requiring
minimal changes to the model architecture. We demonstrate the effectiveness of
our approach on a wide range of benchmarks for natural language understanding.
Our general task-agnostic model outperforms discriminatively trained models that
use architectures specifically crafted for each task, significantly improving upon the
state of the art in 9 out of the 12 tasks studied. For instance, we achieve absolute
improvements of 8.9% on commonsense reasoning (Stories Cloze Test), 5.7% on
question answering (RACE), and 1.5% on textual entailment (MultiNLI).

## Table

* 구현하는 paper에서 제시하는 benchmark dataset을 활용하여 구현하여, 논문에서 제시한 성능과 비교합니다.
  + benchmark dataset은 하나만 골라주세요.
    1. 논문에서 제시한 hyper-parameter와 architecture로 재현을 합니다.
    2. 만약 재현이 안된다면, 본인이 변경한 사항을 서술해주세요.

## Training history

* tensorboard 또는 weights & biases를 이용, 학습의 로그의 스크린샷을 올려주세요.

## OpenAPI로 Inference 하는 방법

* curl ~~~

## Usage

### Environment

* install from source code
* dockerfile 이용

### Training & Evaluate

* interface
  + ArgumentParser의 command가 code block 형태로 들어가야함.
    - single-gpu, multi-gpu

### Inference

* interface
  + ArgumentParser의 command가 code block 형태로 들어가야함.

### Project structure

* 터미널에서 tree커맨드 찍어서 붙이세요.

### License

* Licensed under an MIT license.
