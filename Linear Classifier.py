#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 19:33:40 2020

@author: jonasschroeder
"""

import sklearn.datasets
newsgroups = sklearn.datasets.fetch_20newsgroups_vectorized()

X, y = newsgroups.data, newsgroups.target

X.shape # words appearing in the news articles
y.shape # articles topic

