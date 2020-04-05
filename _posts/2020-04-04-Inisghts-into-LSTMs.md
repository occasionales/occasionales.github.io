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