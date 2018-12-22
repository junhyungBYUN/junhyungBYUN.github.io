---
layout: post
title: Graph-based Semi-Supervised Learning
date: 2018-12-22 23:21:20 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
img: post-7.png # Add image post (optional)
tags: [Blog, Machine Learning]
author: junhyung BYUN # Add name author (optional)
---
<br/>
<br/>

### 안녕하세요, 이번 포스팅에서는 Graph-based Semi-Supervised Learning에 대해 알아보겠습니다.

### 본 내용은 고려대학교 산업경영공학과 강필성 교수님의 강의와 강의자료를 바탕으로 작성됐음을 밝힙니다. 

### 또한, 본 내용은 먼저 이론 내용 전체를 살펴본 뒤 그 이론에 대한 독자의 이해를 바탕으로 Python Code를 살펴보는 방식으로 구성돼 있습니다. 

### 자, 그럼 이제 시작해볼까요?

<br/>

# Semi-Supervised Learning

### Graph-based Semi-Supervised Learning을 알아보기 전에, 먼저 Semi-Supervised Learning이란 무엇일까요?

### Semi-Supervised Learning이란 Machine Learning(기계학습)의 한 학습방법입니다. 

### 그리고 기계학습의 다양한 학습방법 중 대표적인 세 가지 방법을 들면, 아래의 그림과 같이 Supervised Learning과 Unsupervised Learning 그리고 Semi-Supervised Learning을 들 수 있습니다. 

![GbSSL_01]({{site.baseurl}}/assets/img/GbSSL_01.png)

### 기계학습에 필요한 Data는 x변수(독립변수 또는 예측변수 또는 입력변수) 또는 y변수(종속변수 또는 반응변수 또는 출력변수)로 이루어져 있습니다. 

### 각 학습방법을 구분하는 기준은 기계학습을 적용하기 위한 Data의 구성입니다. 

### y변수의 instance가 x변수의 instance 개수만큼 전부 존재하는지, 전부 존재하지 않는지 아니면 일부만 존재하는지에 따라 적용할 수 있는 기계학습 방법이 다릅니다. 

### 그리고 각각의 기준으로 Supervised, Unsupervised, Semi-Supervised Learning으로 나뉩니다. 

### 이때, y변수는 연속형 실숫값이 될 수도 있고 범주형 값이 될 수도 있지만, 본 내용에서는 범주형 값(class label)을 기준으로 설명하겠습니다.
<br/>

### Supervised Learning은 label 값들을 전부 알고 있는 상태에서 사용자가 가지고 있는 Data를 충분히 잘 설명하면서 새로운 Data(x변수)에 대해서도 잘 labeling(y instance) 할 수 있는 y=f(x) 모델을 학습하는 방법입니다.

### Unsupervised Learning은 label을 전부 모르기 때문에, 사용자가 가지고 있는 Data의 내재적인 속성을 학습하여 label을 찾을 때 사용하기도 합니다. 

### 마지막으로 Semi-Supervised Learning은 Supervised Learning처럼 label이 없는 새로운 Data에 대해서 잘 labeling 하는 y=f(x) 모델을 학습하는 방법입니다만, 학습할 때 label이 없는 기존의 Data도 Input으로 함께 활용합니다. 

### 그리고 어떻게 활용하면 Supervised Learning의 성능이 향상될지를 고민하게 됩니다. 

### 즉, label이 없는 기존의 Data라도 그 Data들을 학습 과정에 추가하면 무엇인가 조금 더 좋아지지 않을까? 라는 기대에서 시작된 것입니다.

<br/>
<br/>

# Semi-Supervised Learning vs. Transductive Learning
### 앞에서 설명했듯이 Semi-Supervised Learning은 ‘새로운’ Data에 대해서 잘 labeling 하는 것에 관심이 있습니다. 

### 반면에 Transductive Learning은 모델을 학습하는데 사용된 ‘기존’ Data 가운데 label이 없는 Data의 label이 무엇인지에 관심이 있습니다.

![GbSSL_02]({{site.baseurl}}/assets/img/GbSSL_02.png)

### Semi-Supervised Learning처럼 labeling 하는 것에 관심이 있는 것은 같지만, 그 대상이 모델을 학습하는데 사용한 label이 없는 Data, 즉 Unlabeled Training Data라는 것이죠. 

### 이처럼 두 방법은 서로 다른 개념이지만, 많은 사람들이 두 학습방법을 섞어 쓰기 시작하면서 그 경계가 허물어졌다고 합니다. 

### 그럼에도 불구하고 두 방법을 엄밀히 구분하여 설명한 이유는, Graph-based Semi-Supervised Learning이 Transductive Learning이기 때문입니다.

<br/>
<br/>

# Dataset의 Graph Node 화 및 Node 간 Indirect 연결 방식
