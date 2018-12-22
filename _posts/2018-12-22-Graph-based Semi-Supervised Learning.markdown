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
<br/>

# Semi-Supervised Learning

### Graph-based Semi-Supervised Learning을 알아보기 전에, 먼저 Semi-Supervised Learning이란 무엇일까요?

### Semi-Supervised Learning이란 Machine Learning(기계학습)의 한 학습방법입니다. 

### 그리고 기계학습의 다양한 학습방법 중 대표적인 세 가지 방법을 들면, 아래의 그림과 같이 Supervised Learning과 Unsupervised Learning 그리고 Semi-Supervised Learning을 들 수 있습니다. 
<br/>
![GbSSL_01]({{site.baseurl}}/assets/img/GbSSL_01.png)
<br/>

### 기계학습에 필요한 Data는 x변수(독립변수 또는 예측변수 또는 입력변수) 또는 y변수(종속변수 또는 반응변수 또는 출력변수)로 이루어져 있습니다. 

### 각 학습방법을 구분하는 기준은 기계학습을 적용하기 위한 Data의 구조(Structure)입니다. 

### y변수의 instance가 x변수의 instance 개수만큼 전부 존재하는지, 전부 존재하지 않는지, 아니면 일부만 존재하는지에 따라 적용할 수 있는 기계학습 방법이 다릅니다. 

### 그리고 각각의 기준으로 Supervised, Unsupervised, Semi-Supervised Learning으로 나뉩니다. 

### 이때, y변수는 연속형 실숫값이 될 수도 있고 범주형 값이 될 수도 있지만, 본 내용에서는 범주형 값(class label)을 기준으로 설명하겠습니다.

### Supervised Learning은 label 값들을 전부 알고 있는 상태에서 사용자가 가지고 있는 Data를 충분히 잘 설명하면서 새로운 Data(x변수)에 대해서도 잘 labeling(y instance) 할 수 있는 y=f(x) 모델을 학습하는 방법입니다.

### Unsupervised Learning은 label을 전부 모르기 때문에, 사용자가 가지고 있는 Data의 내재적인 속성을 학습하여 label을 찾을 때 사용하기도 합니다. 

### 마지막으로 Semi-Supervised Learning은 Supervised Learning처럼 label이 없는 새로운 Data에 대해서 잘 labeling 하는 y=f(x) 모델을 학습하는 방법입니다만, 학습할 때 label이 없는 기존의 Data도 Input으로 함께 활용합니다. 

### 그리고 어떻게 활용하면 Supervised Learning의 성능이 향상될지를 고민하게 됩니다. 

### 즉, label이 없는 기존의 Data라도 그 Data들을 학습 과정에 추가하면 무엇인가 조금 더 좋아지지 않을까? 라는 기대에서 시작된 것입니다.

---

<br/>
<br/>

# Semi-Supervised Learning vs. Transductive Learning

### 앞에서 설명했듯이 Semi-Supervised Learning은 ‘새로운’ Data에 대해서 잘 labeling 하는 것에 관심이 있습니다. 

### 반면에 Transductive Learning은 모델을 학습하는데 사용된 ‘기존’ Data 가운데 label이 없는 Data의 label이 무엇인지에 관심이 있습니다.

![GbSSL_02]({{site.baseurl}}/assets/img/GbSSL_02.png)

### Semi-Supervised Learning처럼 labeling 하는 것에 관심이 있는 것은 같지만, 그 대상이 모델을 학습하는데 사용한 label이 없는 Data, 즉 Unlabeled Training Data라는 것이죠. 

### 이처럼 두 방법은 서로 다른 개념이지만, 많은 사람들이 두 학습방법을 섞어 쓰기 시작하면서 그 경계가 허물어졌다고 합니다. 

### 그럼에도 불구하고 두 방법을 엄밀히 구분하여 설명한 이유는, Graph-based Semi-Supervised Learning이 Transductive Learning이기 때문입니다.

---

<br/>
<br/>

# Dataset의 Graph Node 화 및 Node 간 Indirect 연결 방식

### 먼저 아래의 그림을 보면,

![GbSSL_03]({{site.baseurl}}/assets/img/GbSSL_03.png)

### 주어진 왼쪽의 Data Set에서 instance 단위로 오른쪽 Graph의 Node에 할당되는 것을 알 수 있습니다. 

### 이는 각각의 instance를 Graph에서 Node로 표현하는 방식임을 알 수 있습니다. 

### 이때, label 값이 있는 instance는 해당 label의 값이 Node에 나타나고 label 값이 없는 instance는 label 값이 없는 빈 Node로 남게 됩니다. 

### 그리고 Node 간 연결된 선들을 Edge라고 하는데, 이는 Node 간 유사도(similarity)를 나타냅니다. 

### 사용하는 Graph의 유사도 조건을 충족해야 Node가 연결되고, 연결된 Node 간 유사도가 클수록 굵은 선으로 나타냅니다. 

### 예를 들어, 이미 label이 있는 x1과 x2는 서로 다른 label이기 때문에 Node 간 연결돼 있지 않습니다. 

### 반면에 label이 없는 x3 Node의 label을 추정하는데, +1 label 값을 갖는 x1 Node로부터 시작해서 굵은 선을 따라 추정하게 됩니다. 

### 하지만 이때, x1 Node에서 x3 Node로 바로 추정하는 것이 아닙니다.

### 먼저 x1 Node의 오른쪽에 연결된 유사도가 큰 Node의 label을 먼저 추정합니다. 

### 그리고 추정한 그 Node의 label을 가지고 굵은 선을 따라 또다시 바로 오른쪽의 Node의 label을 추정합니다.

### 이런 식으로 연속해서 결국 x3 Node의 label을 추정하게 됩니다. 

### 즉, x1 Node에서 x3로 한 번에 label을 추정하긴 어렵지만, 바로 옆의 유사한 Node를 통해 한 다리씩 건너서 추정하면(Indirect) 알고 싶은 label을 추정할 수 있습니다. 

### 따라서 유사도는 Direct로 추정하기보다는 기본적으로 Indirect 방법으로 추정한다는 것을 알 수 있습니다. 

### 그리고 label을 전파한다고 해서 label propagation으로 부르기도 합니다. 

### 이와 관련된 예시는 아래의 손글씨 인식 그림을 통해서도 확인할 수 있습니다.

![GbSSL_04]({{site.baseurl}}/assets/img/GbSSL_04.png)

### 이처럼 Graph-based Learning의 가정은 두 Node가 유사하면(Edge 굵기가 굵으면, heavy Edge) 근처의 유사한 Node의 label과 동일한 label을 갖는 것이 합당하다는 것입니다. 

### 이를 통해 기존의 Unlabeled Data의 Label이 무엇인지(Transductive Learning)를 추정하게 됩니다. 

---

<br/>
<br/>

# 수학적 기호 정의와 Graph 종류 및 유사도 가중치 계산방식

### 그렇다면 실제로 Unlabeled Data에 대해 label을 찾기 위한 수학적 알고리즘은 어떻게 될까요?

### 이를 위해 우선 Graph에서의 Node와 Edge를 위한 수학적 정의를 먼저 내리게 됩니다.

![GbSSL_05]({{site.baseurl}}/assets/img/GbSSL_05.png)


![GbSSL_05_2]({{site.baseurl}}/assets/img/GbSSL_05_2.png)


![GbSSL_06]({{site.baseurl}}/assets/img/GbSSL_06.png)

---

<br/>
<br/>

# Graph-based SSL에서 수학적으로 Label을 추정하는 방법1

### 우리가 알고 싶은 것은 기존에 가지고 있는 Unlabeled Data에 대한 label 값입니다. 

### 앞에서 설명했듯이, Graph-based Semi-Supervised Learning의 label 추정 방법은 항상 기존에 알고 있는 label 값으로 추정하기 시작합니다. 

### 따라서 기존 label과 추정 label로 각각 part를 나누어 생각해볼 수 있고 이때 minimum cut algorithm을 사용합니다. 

### minimum cut algorithm이란 Network Flow에서 등장하는 개념으로, 줄여서 mincut algorithm으로 부르며 네트워크식으로 연결된 Node들의 Edge에 따라 비용을 최소화하기 위해 Node들의 간선을 cut하는 방식입니다. 

### 이때, Graph의 방식이 Direct인지 Indirect인지에 따라 cut하는 방식이 달라집니다. 

### Graph-based Semi-Supervised의 경우 Indirect Graph 방식이기 때문에 cut했을 때의 비용으로, cut하는 단순 간선의 개수가 될 수 있고, cut하는 간선의 가중치(Edge)의 합이 될 수도 있습니다. 

### 본 내용의 경우, 유사도를 가중치로 계산하기 때문에 가중치의 합이 최소가 되는 방향으로 cut하는 Graph가 될 것을 알 수 있습니다. 

### 조금 더 이해를 돕기 위해 아래의 그림 예시를 참고하시면 될 것 같습니다.

![GbSSL_07]({{site.baseurl}}/assets/img/GbSSL_07.png)

### 그리고 이를 수식화하면,

![GbSSL_08]({{site.baseurl}}/assets/img/GbSSL_08.png)

### 이 되고 우선 실제 label 값 y_l을 고정합니다.

### 이 말은 label을 알고 있는 Data는 항상 완벽해서 Noise나 Error가 전혀 없다고 보기 때문에, 기존 label은 반드시 유지해야 한다는 것을 의미합니다.

### 그러면서 추정하고 싶은 label에 대해서는 0 또는 1의 정수만 가질 수 있는 것도 의미합니다.

### 즉, 0~1 사이의 실숫값도 허용하지 않고 solution 자체가 0 아니면 1이어야 하는, 엄격한 기준을 세우고 있는 것입니다. 

### 이렇게 두 가지 전제 조건 아래에, 유사도 w를 이용한 추정 label의 오차 절대 합이 최소가 되게 하는 것이 목적입니다.

### 만약 유사도 w가 매우 작을 때는 y_i와 y_j의 값의 큰 차이 때문에 비용을 최소화하기 위해 Graph 상에서 cut를 진행하게 됩니다. 

### 즉, 이를 최적화 문제로 바꾸어 해결하면, 

![GbSSL_09]({{site.baseurl}}/assets/img/GbSSL_09.png)

### 로 바꿔서 풀 수 있습니다.

### 그리고 이때 위의 그림대로 최소화할 각 항을 1번 part와 2번 part로 나누어 생각해볼 수 있습니다.

### 왜냐하면, 각각 기존 label과 추정 label에 해당하기 때문입니다.

### 우선, 1번 part에 해당하는 ‘기존 label part’는 추정 label y_i와 정답 label y_li가 모두 같으면 0, 하나라도 다르면 ∞ penalty를 부여하는 것을 의미합니다.

### 즉, 추정된 label과 정답 label이 조금이라도 다른 것을 허용하지 않겠다는 뜻입니다.

### 앞에서 기존 label y_l은 모두 완벽하다고 보고 그 값들을 모두 고정한 것과 일맥상통하는 대목입니다.

### 다음으로 2번 part에 해당하는 ‘추정 label part’는 서로 근처에 있고 유사한 Node들이면 그 Node들 사이의 label은, 최대한 같은 값이 되도록 만들라는 뜻입니다.

### 유사도가 클수록 근처의 Node들은 label이 서로 같아야 한다고 보는 Graph-based Learning의 가정을 담고 있다고 할 수 있습니다.

### 물론 이때, y_i 또는 y_j가 가질 수 있는 값은 0과 1뿐입니다.

---

<br/>
<br/>

# Graph-based SSL에서 수학적으로 Label을 추정하는 방법2

![GbSSL_11]({{site.baseurl}}/assets/img/GbSSL_11.png)

### 이번에는 Label을 추정하는 방법1에서 ‘추정 label part’에 해당하는 조건을 완화하여 label을 추정하는 방법입니다.

### 즉, 추정하려는 label이 항상 0 아니면 1이어야 한다는 정수 가정을 실수 가정으로 바꿔 완화하는 방법입니다.

### 이를 위해, Harmonic function f(x_i)=y_i를 사용합니다. 즉, 

![GbSSL_12]({{site.baseurl}}/assets/img/GbSSL_12.png)

### 으로 바뀌게 됩니다. 

### 두 Node i와 j의 label이 항상 0 또는 1은 아니어도 되지만, 유사도가 높을수록 Harmonic function 값이 반영된 f(x_i)실숫값과 f(x_j)실숫값이 서로 유사해야 합니다. 

### 그리고 나중에 실제 label을 달 때는 cut-off를 정해서 그 기준으로 label을 확정 짓게 됩니다. 

### 그런데, 이 ‘추정 label part’에 해당하는 계산과정에서 Graph Laplacian Matrix를 사용해서 비교적 간단하게 계산하는 방법을 사용하게 됩니다.

---

<br/>
<br/>

# Graph Laplacian Matrix

### label이 있는 Data의 개수는 l이고 label 없는 Data의 개수는 u이므로 모든 Data의 개수는 l+u가 됩니다.

### 지금부터 설명할 대부분의 Matrix는 행 또는 열이 모두 l+u인 행렬들로, label을 추정하는 두 번째 방법에서, ‘추정 label part’에 해당하는 식을 간단하게 계산하기 위해 다음 과정의 행렬들을 계산합니다.

![GbSSL_13]({{site.baseurl}}/assets/img/GbSSL_13.png)

### 즉, Graph Laplacian Matrix를 구해서 ‘추정 label part’를 계산하면,

![GbSSL_14]({{site.baseurl}}/assets/img/GbSSL_14.png)

### 이 됩니다.

### 예를 들어 아래의 Graph(Node의 숫자는 각 Node의 label이 아닌 Node 번호)를 갖는 경우,

![GbSSL_15]({{site.baseurl}}/assets/img/GbSSL_15.png)

### 먼저 유사도 행렬 W를 Node간 연결이 된 경우에는 1, 그렇지 않으면 0인 방식(Adjacency Matrix)으로 계산합니다.

### 이어서 이 W의 열별 유사도의 합을 계산하여 Diagonal Degree Matrix D를 구하여 최종적으로 Graph Laplacian Matrix를 계산합니다.

### 이때, ‘추정 label part’ 식은

![GbSSL_16]({{site.baseurl}}/assets/img/GbSSL_16.png)

### 으로 표현할 수 있는 것을 확인할 수 있습니다.

### 한편, ‘추정 label part’의 Graph Laplacian Matrix를 이용해서 실제로 우리가 알고 싶은 label의 추정값(f_u) 부분만을 쉽게 계산할 수 있습니다.

### 바로 Partition Laplacian Matrix를 사용하는 것인데요, 이제부터 살펴보겠습니다. 

![GbSSL_17]({{site.baseurl}}/assets/img/GbSSL_17.png)

### 그리고 Partition Laplacian를 적용하기 위해 ‘추정 label part’를 Harmonic Function Vector f에 대해 미분하면,

![GbSSL_18]({{site.baseurl}}/assets/img/GbSSL_18.png)

---

<br/>
<br/>

# Graph-based SSL에서 수학적으로 Label을 추정하는 방법3

### Graph-based SSL에서 수학적으로 Label을 추정하는 방법1과 방법2에서는 모두 ‘기존 label part’에서 실제 label을 그대로 적용했습니다.

### 그런데 사람들은 실제 Data에 Noise가 있을 수 있기 때문에, 주어진 label이 항상 옳다고 할 수 있는지에 대해 의문을 갖기 시작했습니다.

### 따라서 ‘기존 label part’의 ∞ penalty 조건을 완화하기 시작합니다.

### 일부 주어진 label들이 틀린 label이면 그것을 보존하는 것보다는 틀림을 인지하고 수정하는 것이 더 좋은 결과물이 될 수 있다고 보기 때문입니다.

### 하지만, 제대로 된 기존 label을 바꾸는 것은 막아야 하므로 penalty 장치를 두게 되고 그 식은 다음과 같습니다.

![GbSSL_19]({{site.baseurl}}/assets/img/GbSSL_19.png)

### 다시 말해서, ‘기존 label part’에서 ∞가 사라져 조건을 완화한 대신, 제대로 된 기존 label을 바꾸는 것은 막기 위해 penalty λ를 적용하게 됩니다.

### 이때, λ는 크게 할수록 실제 labeled Data의 label은 변할 가능성이 커지고 작게 할수록 실제 label을 보존하는 방향으로 설정됩니다.

### 그리고 이 방법이 가장 현실적으로 생각하고 적용할 수 있는 Graph-based Semi-Supervised Learning 방법이며 이에 대한 solution은 아래와 같습니다.

![GbSSL_20]({{site.baseurl}}/assets/img/GbSSL_20.png)

### 즉, Graph-based Semi-Supervised Learning에서는 기존에 가지고 있는 Data의 label을 추정하는데 수학적으로 항상 명시적인 해(explicit solution)가 존재한다는 것을 알 수 있습니다.

### 이제부터는 이러한 이론적 배경을 기반으로 Graph-based Semi-Supervised Learning을 Python으로 구현한 Code를 살펴보겠습니다.

---

<br/>
<br/>

# Python Code for Graph-based Semi-Supervised Learning
```{.python}
import os
import numpy as np
import numpy.linalg as lin
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse 
from scipy.sparse.linalg import inv
from scipy.spatial import distance
```

```{.python}
def e_radius(euc, epsilon):
    if epsilon <= 0:
        print('Use epsilon >= 0')
        return None
    e_distance = np.where(euc < epsilon, euc, np.inf)
    return e_distance
```    
