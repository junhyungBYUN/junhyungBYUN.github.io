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

안녕하세요, 이번 포스팅에서는 Graph-based Semi-Supervised Learning에 대해 알아보겠습니다. 

본 내용은 고려대학교 산업경영공학과 강필성 교수님의 강의와 강의자료를 바탕으로 작성됐음을 밝힙니다. 

또한, 본 내용은 먼저 이론 내용 전체를 살펴본 뒤 그 이론에 대한 독자의 이해를 바탕으로 Python Code를 살펴보는 방식으로 구성돼 있습니다. 

자, 그럼 이제 시작해볼까요?

<br/>

# Semi-Supervised Learning

Graph-based Semi-Supervised Learning을 알아보기 전에, 먼저 Semi-Supervised Learning이란 무엇일까요?

Semi-Supervised Learning이란 Machine Learning(기계학습)의 한 학습방법입니다. 

그리고 기계학습의 다양한 학습방법 중 대표적인 세 가지 방법을 들면, 아래의 그림과 같이 Supervised Learning과 Unsupervised Learning 그리고 Semi-Supervised Learning을 들 수 있습니다. 

![GbSSL_01]({{site.baseurl}}/assets/img/GbSSL_01.png)

기계학습에 필요한 Data는 변수(독립변수 또는 예측변수 또는 입력변수) 또는 변수(종속변수 또는 반응변수 또는 출력변수)로 이루어져 있습니다. 

각 학습방법을 구분하는 기준은 기계학습을 적용하기 위한 Data의 구성입니다. 변수의 instance가 변수의 instance 개수만큼 전부 존재하는지, 전부 존재하지 않는지 아니면 일부만 존재하는지에 따라 적용할 수 있는 기계학습 방법이 다릅니다. 

그리고 각각의 기준으로 Supervised, Unsupervised, Semi-Supervised Learning으로 나뉩니다. 

이때, 변수는 연속형 실숫값이 될 수도 있고 범주형 값이 될 수도 있지만, 본 내용에서는 범주형 값(class label)을 기준으로 설명하겠습니다.
