---
layout: post
title: Locating Objects without bounding boxes (Ribera et al., 2019)
date: 2020-11-09 08:39
author: Alex To
tags: deep-learning computer-vision convolutional-neural-network hausdorff-distance
comments: true 
---

> In this paper , the authors proposed a loss function based on the average Hausdorff distance between two unordered sets of points. The proposed method has no notion of bounding boxes, region proposals or sliding windows.

<!--more-->

{% cite Ribera2019 %} aim to achieve object localization without using bounding boxes. That means, instead of annotating the images with boxes, the ground truth labels are now points. The authors argued that using points as ground truth labels might be less laborious to obtain in some cases where bounding boxes are not required.

Supposed we have ground truth labels as a set of points, let's call it $$A$$, we want to estimate a set $$B$$ that is as close to $$A$$ as possible. To train the network, we need to measure how far off the estimated set B is from A. We all know how to measure the distance between two points right?, but how to measure distance between two sets of points ? One way is to built on the idea of Hausdorff Distance. If you are not familiar with what Hausdorff Distance is, you may want to check out my previous blog post [here]({% post_url 2020-11-09-hausdorff-distance %}) ;)

Alright, so now we know that we can build a loss function based on Average Hausdorff Distance (AHD), let's see why it might be tricky to construct a loss function from its original form. The AHD is defined by

$$
d_{AH}(A, B) = \frac{1}{|A|}\sum_{a \in A}\min_{b \in B}d(a, b) + \frac{1}{|B|}\sum_{b \in B}\min_{a \in A}d(a, b)
$$

Note that in the paper, the authors used the notion of $$X$$ and $$Y$$ but I replaced them with $$A$$ and $$B$$ to avoid confusion with $$(x, y)$$ coordinates used later in this post.   

Now we have to think about the outputs of the network. Ideally, if we want to use $$d_{AH}$$ as the loss function, we want the network to output a set of coordinates $$(x, y)$$, perhaps, at the final linear layer as illustrated in Figure 1.

{% figure caption:"Figure 1" class:"width_500"%}
![](/assets/images/nobboxes_1.png)
{% endfigure %}

The issue is that the size of the final linear layer is fixed, so the estimated number of points is fixed. This is not desirable because the number of points is different from image to image so let's not do that. Another way is to let the network output a heatmap or probability map $$P$$ where $$p_{x, y}$$ is the probability that $$(x, y)$$ is a key point similar to {% cite Ronneberger2015 %} as illustrated in Figure 2.

{% figure caption:"Figure 2" class:"width_500"%}
![](/assets/images/nobboxes_2.png)
{% endfigure %}

The new output can predict any number of estimated points, but it no longer returns pixel coordinates. Hence, we need a bit of modification to the original AHD so that the loss function is differentiable with respect to the output $$P$$.

The authors proposed to replace AHD with an approximation  

$$
d_{WH}(p, Y) = \frac{1}{S + \epsilon}\sum_{a \in \Omega}p_a\min_{b \in B}d(a, b) + \frac{1}{|B|}\sum_{b \in B} \underset{a \in \Omega}{M_{\alpha}}[p_ad(a, b) + (1-p_a)d_{max}]
$$

where

$$
S = \sum_{x \in \Omega}p_x
$$

$$
\underset{a \in \Omega}{M_{\alpha}}[f(a)] = \left(\frac{1}{|A|}\sum_{a \in A}f^{\alpha}(a)\right)^{\frac{1}{\alpha}}
$$

Quite a lot of things going on here so let's break it down bit by bit. 

Let  

$$U = \frac{1}{S + \epsilon}\sum_{a \in \Omega}p_a\min_{b \in B}d(a, b)$$ 

$$V = \frac{1}{|B|}\sum_{b \in B} \underset{a \in \Omega}{M_{\alpha}}[p_ad(a, b) + (1-p_a)d_{max}] $$

First of all, the intuition of this loss function is that if a predicted point $$a$$ is far from set $$B$$ then $$p_a$$ must be low. But wait, why do we need both $$U$$ and $$V$$ ? Well, if $$V$$ is absent then the model will predict an empty set $$A$$ by setting all $$p_a$$ to 0, that will minimize the cost to 0. Similarly if $$U$$ is absent then the model will predict the entire image as set $$A$$, that will also minimize the cost to 0 as the image contains set $$B$$ so the distance between the image and set $$B$$ is 0. Both terms must be there so that the model won't predict too few or too many points than expected. 

{% figure caption:"Figure 3" class:"width_500" %}
![](/assets/images/nobboxes_3.png)
{% endfigure %}

$$U$$ is an approximation to 
$$ \frac{1}{|A|}\sum_{a \in A}\min_{b \in B}d(a, b)$$ so you might think we can do the same with $$V$$ by letting 

$$V = \frac{1}{B}\sum_{b \in B}\min_{b \in B}p_ad(a, b)$$

However, unlike $$U$$, now $$p_a$$ appears inside the $$\min$$ which makes the loss function not differentiable with respect to $$p$$. A good approximation to the minimum function is the [generalized mean](https://en.wikipedia.org/wiki/Generalized_mean). You may find some good discussion about it [here](https://mathoverflow.net/questions/35191/a-differentiable-approximation-to-the-minimum-function) on Mathoverflow.

One final thing to note is that the authors added a Smooth L1 Loss for the regression of the object count so the final loss becomes

$$ \mathcal{L}(p, Y) =  d_{WH}(p, Y) + \mathcal{L}_{reg}(C - \hat{C}) $$

where $$C$$ is the number of target points and $$\hat{C}$$ is a scalar output from the network predicting the estimated object count. 

Hope you like my post and if there is any part that is not clear, leave a comment below, so I can update this post with more details ;) 

References
---

{% bibliography --cited%}


