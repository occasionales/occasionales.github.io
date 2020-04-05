---
layout: post
title: Insights into LSTM architecture
subtitle: Finding the total number of mutliply and accumulate operations
image: /img/LSTMcell-1.png
gh-repo: occasionales/occasionales.github.io
gh-badge: [follow]
tags: [test]
comments: true
---

While working on my master thesis, I had to look at various LSTM networks and quickly try to estimate how well they would be suited for embedded implementation. That is, would these networks be an excellent candidate to put on a microprocessor with a limited amount of computing resources available. I therefore looked mainly at three things

* Number of parameters in network.
* Feature map size.
* Number of multiply and accumulate operations (MACC).

I'll now go more deeply into how to calculate two of these considerations, in particular number of parameters and number of MACCs. The feature map size depends very highly on how the feature map are stored on the RAM, but very rough approach is to take the two largest consecutive feature maps and calculate the size of those together. First let us look a bit into LSTMs.

##Long Short-Term Memory
Recurrent neural networks are a class of neural networks, where node connections form a directed graph along a temporal sequence. In more simple terms, they are networks with loops that allow information to endure within the network.

| ![Typical RNN network structure with RNN cells](/img/RNN.png){: .center-block :} | 
|:--:| 
| *Typical RNN network structure with RNN cells, here _X_<sub>_t_</sub>: input time sequence, _h_<sub>_t_</sub>: output time sequence.* |

A problem early versions of RNNs suffer from is the vanishing gradient problem~\cite{hochreiter_vanishing_1998}, and this led to gradient-based methods having an extremely long learning time in training RNNs. This was because the error gradient, for which gradient-based methods need, vanishes as it gets propagated back through the network. This leads to layers in RNNs, usually the first layers, to stop learning. Therefore when a sequence is long enough, RNNs struggle to propagate information from earlier time steps to later ones. In simpler terms, RNNs suffer from short term memory. By looking at the figure above, a typical problem early versions of RNNs faced is that cell one would receive a vanishing error gradient because of the decaying error back-flow and, therefore, not be able to propagate the correct information to cell 4. To combat this short term memory, Sepp Hochreiter and Jürgen Schmidhuber introduced a novel type of RNN called long short-term memory (LSTM). The structure of the LSTM has changed over the years, and here a description of the most common architecture will follow. An LSTM unit is composed of a cell, and within the cell, three gates control the flow of information within the LSTM cell and control the cell state. The three gates are: an input gate, an output gate, and a forget gate. LSTM then chains together these cells, where each cell within the LSTM serves as a memory module.

