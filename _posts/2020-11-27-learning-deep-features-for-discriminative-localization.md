---
layout: post
comments: true
title: Learning deep features for discriminative localization
date: 2020-11-27 22:20
category: 
author: Alex To
tags: cam deep-learning computer-vision convolutional-neural-network
summary: 
---

> In this paper, the authors discussed how Global Average Pooling enables localization ability despite being trained on image-level labels i.e without bounding boxes.

<!--more-->

The main motivation of this paper {% cite Zhou2016 %} is how to generate decision heatmap or class activation maps (CAM) in CNNs. Simply put, if the CNN model classifies an image with label Y, we would like to know which pixels from the original image trigger such decision (Figure 1). 

{% figure caption:"Figure 1. Image taken from the paper" class:"width_500" %}
![](/assets/images/cam-1.png)
{% endfigure %}

One observation is that the convo layers actually behave as object detectors with deeper layers learn more abstract concepts. Despite not being train with bounding boxes, the convo layers actually have a remarkable ability to localize objects but this ability is lost when fully connected layers are used in the last few layers in traditional CNNs.  [Global Average Pooling]({% post_url 2020-11-21-global-average-pooling %}) was introduced in {% cite Lin2014 %} to mitigate this problem. Removing FCs layers also reduces the number of parameters, thus, might reduce over-fitting.

To use CAM, modification must be made to existing networks so that the <mark> final convo is followed by GAP and then a final FC layer. </mark>

> For a given image, let $$ f_k(x, y) $$ represent the activation of unit k in the last convolutional layer at spatial location $$(x, y)$$. Then, for unit $$k$$, the result of performing global average pooling, $$F^k$$ is $$\sum_{x,y}f_k(x, y)$$

{% figure caption:"Figure 2" %}
![](/assets/images/cam-2.png)
{% endfigure %}

As illustrated in Figure 2, $$f_k(x, y)$$ is the single value at $$(x, y)$$ at the feature map $$k$$. Since the convo layer is followed by GAP, $$F^k$$ is the average of all $$f_k(x,y)$$. We could also defined $$F^k$$ as the sum of all $$f_k(x, y)$$ as average is just the sum divided by a constant.

As mentioned earlier, after GAP, we add a final FC layer for classification. Let $$S_c$$ be the activation for class $$c$$ before softmax. Then $$S_c$$ is just the dot product between the vector $$F$$ and the weight vector for $$c$$

$$
S_c = F \cdot w^c = \sum_{k}w^c_kF^k
$$

So far so good, we have established the "link" between $$S_c$$ and vector $$F$$. Let's trace back one layer before $$F$$ and establish the "link" between $$S_c$$ and $$(x, y)$$. If we "extract" feature at location $$(x, y)$$ for every feature map to construct a vector $$f$$ as shown in Figure 3, then notice that $$f$$ has the same length as $$F$$ and $$w^c$$, obviously, due to GAP ;)

{% figure caption:"Figure 3" %}
![](/assets/images/cam-3.png)
{% endfigure %}

We define, $$M_c(x, y)$$, the importance of $$(x, y)$$ as the dot product between $$f$$ and $$w^c$$

$$
M_c(x,y) = \sum_kw^c_kf_k(x,y)
$$

Hence,

$$
S_c = \sum_{x,y}M_c(x,y)
$$

Since, $$S_c$$ is the sum of all $$M_c(x, y)$$ then the larger $$M_c(x, y)$$ is, the more it contributes to $$S_c$$ that leads to the classification of an image to class $$c$$.

Finally, we upscale $$M_c$$ to the original size of the image to get the class activation map ;)

{% bibliography --cited %}
