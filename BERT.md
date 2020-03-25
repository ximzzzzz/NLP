## BERT(Bidirectional Encoder Representations from transformer)



트랜스포머 기반 모델로 GPT, ELMo 와 다르게 전 레이어에서 양방향으로 학습함

기존 ReLU 대신 GELU(Gaussian Error Linear Units)를 사용함(ReLU에 비해 0주변에서 부드러운 곡선)

encoder 내에 residual connection & normalization 삽입



### 학습데이터 

​	마스크언어모델(Masked LM) 용 학습데이터

- 학습데이터 한 문장 토큰의 15%를 마스킹
- 마스킹 대상 토큰 중 80%는 실제 빈칸으로 만들고 모델은 그 빈칸을 채운다.
- 마스킹 대상 토큰 중 10%는 랜덤으로 다른 토큰으로 대체하고 모델은 해당 위치의 정답이 무엇일지 맞춘다.
- 마스킹 대상 토큰 중 10%는 토큰 그대로 두고 모델은 정답이 무엇일지 맞춘다.

​	

​	위 학습데이터를 통해 다음 효과를 노릴 수 있다.

> 마스킹된 빈칸을 채워야 하기 때문에 문장 내 어느 자리에 어떤 단어를 쓰는지 문맥을 학습하게 된다.
>
> 다른토큰으로 대체된것과 그대로 둔 문장을 통해 학습하며 주어진 문장이 의미/문법상 비문인지 학습한다
>
> 모델은 어떤 단어가 마스킹 될 지 모르기 때문에 문장 내 모든 단어 사이의 관계를 학습한다.



​	다음문장예측(Next Sentence Prediction)

- 학습데이터는 1건당 두 문장으로 구성된다.

- 전체 학습데이터의 50%는 동일한 문서에서 실제 이어지는 문장을 두개 뽑고 True labelling

- 전체 학습데이터의 50%는 서로 다른 문서에서 문장을 뽑은 뒤 False labelling

- `max_num_tokens`를 정의하여 학습데이터의 90%는 `max_num_tokens`가 사용자가 정한 `max_sequence_length`가

  되도록 한다.

- 나머지 10%는 `max_num_tokens`가 `max_sequence_length`보다 짧게 되도록 랜덤으로 정한다.

- 위에서 뽑은 두 문장의 단어 총 수가 `max_num_tokens`를 넘지 못할때까지 두 문장중 단어가 많은쪽을 50% 확률로

  맨앞 또는 맨뒤 단어 하나씩 제거한다.



​	위 학습데이터를 통해 다음 효과를 노릴 수 있다.

> 두 문장이 이어진 문장인지 학습하며 문장 간 의미/문맥을 학습한다.
>
> 맨 앞/ 맨 뒤 단어가 삭제된 채로 학습했기 때문에 일부 문장성분 없이도 의미를 예측할 수 있다.
>
> 학습데이터의 10%는 `max_sequence_length`보다 짧게 만들었기 때문에 짧은 문장도 커버할 수 있다.



### Architecture

​	encoder

​		self-attention -> multi-head attention -> feed foward network

​		residual connection & normalization



​	decoder

​		

### Fine-tuning BERT

