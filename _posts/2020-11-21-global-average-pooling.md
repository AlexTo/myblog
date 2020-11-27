---
layout: post
comments: true
title: Global average pooling
date: 2020-11-21 14:25
category: 
author: Alex To
tags: deep-learning computer-vision convolutional-neural-network
summary: 
---

> Conventional CNNs perform convolution in the lower layers of the network. For classification, the feature maps of the last convolutional layer are vectorized and fed into fully connected layers followed by a softmax layer. Global average pool is another strategy first proposed by (Lin et al., 2014)  to replace the traditional fully connected layers in CNNs.

<!--more-->

First, let's have a look at a traditional CNN setup such as LeNet. The last two layers are the good old fully connected layers (FCs) before being connected to the final softmax layer.

{% figure caption:"Figure 1. CNN with fully connected layers" class:"width_500" %}
![](/assets/images/lenet.png)
{% endfigure %}

Okay, but FCs are prone to over-fitting so we could either apply drop-out {% cite hinton2012improving %} or get rid of FCs altogether with average pooling after the last convolutional layer {% cite Lin2014 %} as illustrated in Figure 2.

{% figure caption:"Figure 2. CNN with global average pooling" class:"width_500" %}
![](/assets/images/lenet-gap.png)
{% endfigure %}

And that's the whole idea about global average pooling, simple isn't it ? ;)

{% bibliography --cited %}
