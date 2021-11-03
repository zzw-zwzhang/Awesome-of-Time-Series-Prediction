# Awesome-of-Time-Series-Prediction [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

![](https://img.shields.io/badge/Number-6-green)

A curated list of time series prediction and related resources.

Please feel free to pull requests or open an issue to add papers.


### :high_brightness: Updated 2021-10-30


## Table of Contents

- [Type of Time Series Prediction](#type-oof-time-series-prediction)

- [Time Series Prediction](#time-series-prediction)
  - [2021 Venues](#2021)
  - [2020 Venues](#2020)
  - [2019 Venues](#2019)
  - [2018 Venues](#2018)
  - [2017 Venues](#2017)
  - [2016 Venues](#2016)
  - [Previous Venues](#2010-2014)
  - [arXiv](#arxiv)
 
- [Awesome Surveys](#awesome-surveys)

- [Awesome Blogs](#awesome-blogs)

- [Awesome Datasets](#awesome-datasets)

- [Awesome Code](#awesome-code)


### Type of Time Series Prediction

| Type        | `S`          | `MS`           | `Other-driven`                   | `Multi Modal`                 | ``                  | ``              | `Other`     |
|:----------- |:-------------:|:--------------:|:----------------------: |:---------------------:|:----------------------:|:-----------------:|:-----------:|
| Explanation | Stock | Multi-Stock | OD | MM |  |  | other types |



### 2021

| Title    | Venue    | Type     | Code     | Star     |
|:-------- |:--------:|:--------:|:--------:|:--------:|
| [Hierarchical Adaptive Temporal-Relational Modeling for Stock Trend Prediction](https://www.ijcai.org/proceedings/2021/0508.pdf) | IJCAI | ``     | [PyTorch(Author)]()   |  ``  |
| [Bayesian Temporal Factorization for Multidimensional Time Series Prediction](https://arxiv.org/pdf/1910.06366.pdf) | TPAMI | ``     | [PyTorch(Author)]()   |  ``  |
| [Discrete Graph Structure Learning for Forecasting Multiple Time Series](https://arxiv.org/pdf/2101.06861.pdf) | ICLR | `Other`     | [PyTorch(Author)](https://github.com/chaoshangcs/GTS)   |  `67`  |
| [A Study of Joint Graph Inference and Forecasting](https://arxiv.org/pdf/2109.04979.pdf) | ICML-W | ``     | [PyTorch(Author)]()   |  ``  |
| [REST: Relational Event-driven Stock Trend Forecasting](https://arxiv.org/pdf/2102.07372.pdf) | WWW | `MS & OD`     | [PyTorch(Author)]()   |  ``  |
| [Financial time series forecasting with multi-modality graph neural network](https://www.sciencedirect.com/science/article/abs/pii/S003132032100399X) | PR | `MM`     | [PyTorch(Author)]()   |  ``  |
| [End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series](http://proceedings.mlr.press/v139/rangapuram21a/rangapuram21a.pdf) | ICML | ``     | [MexNet(Author)](https://github.com/rshyamsundar/gluonts-hierarchical-ICML-2021)   |  `9`  |


### 2020

| Title    | Venue    | Type     | Code     | Star     |
|:-------- |:--------:|:--------:|:--------:|:--------:|
| [Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations](https://aclanthology.org/2020.emnlp-main.676.pdf) | EMNLP | `OD`     | [PyTorch(Author)](https://github.com/midas-research/man-sf-emnlp)   |  `6`  |
| [Multi-Graph Convolutional Network for Relationship-Driven Stock Movement Prediction](https://arxiv.org/pdf/2005.04955.pdf) | ICPR | `OD`     | [PyTorch(Author)](https://github.com/start2020/Multi-GCGRU)   |  `4`  |
| [Hierarchical Multi-Scale Gaussian Transformer for Stock Movement Prediction](https://www.ijcai.org/proceedings/2020/0640.pdf) | IJCAI | ``     | [PyTorch(Author)]()   |  ``  |
| [Multi-scale Two-way Deep Neural Network for Stock Trend Prediction](https://www.ijcai.org/proceedings/2020/0628.pdf) | IJCAI-FinTech | ``     | [PyTorch(Author)](https://github.com/marscrazy/MTDNN)   |  `30`  |
| [Modeling the Stock Relation with Graph Network for Overnight Stock Movement Prediction](https://www.ijcai.org/proceedings/2020/0626.pdf) | IJCAI-FinTech | `MS`     | [PyTorch(Author)](https://github.com/liweitj47/overnight-stock-movement-prediction)   |  `17`  |
| [A Quantum-inspired Entropic Kernel for Multiple Financial Time Series Analysis](https://www.ijcai.org/proceedings/2020/0614.pdf) | IJCAI-FinTech | `MS`     | [PyTorch(Author)]()   |  ``  |
| [Domain Adaptive Multi-Modality Neural Attention Network for Financial Forecasting](https://github.com/zzw-zwzhang/Awesome-of-Time-Series-Prediction/blob/main/paper/2020%20WWW__Domain%20Adaptive%20Multi-Modality%20Neural%20Attention%20Network%20for%20Financial%20Forecasting.pdf) | WWW | ``     | [PyTorch(Author)]()   |  ``  |
| [Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations](https://aclanthology.org/2020.emnlp-main.676.pdf) | EMNLP | ``     | [PyTorch(3rd)](https://github.com/pigman0315/MAN-SF)   |  `0`  |
| [Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting](https://arxiv.org/pdf/2103.07719.pdf) | NeurIPS | `Other`     | [PyTorch(Author)](https://github.com/microsoft/StemGNN/)   |  `150`  |
| []() | NeurIPS | ``     | [PyTorch(Author)]()   |  ``  |


### 2019

| Title    | Venue    | Type     | Code     | Star     |
|:-------- |:--------:|:--------:|:--------:|:--------:|
| [Multi-task Recurrent Neural Networks and Higher-order Markov Random Fields for Stock Price Movement Prediction](https://songdj.github.io/publication/kdd-19-b/kdd-19-b.pdf) | KDD | `MS`     | [PyTorch(Author)]()   |  ``  |
| [Investment Behaviors Can Tell What Inside: Exploring Stock Intrinsic Properties for Stock Trend Prediction](https://www.microsoft.com/en-us/research/uploads/prod/2019/11/p2376-chen.pdf) | KDD | ``     | [PyTorch(Author)]()   |  ``  |
| [Individualized Indicator for All: Stock-wise Technical Indicator Optimization with Stock Embedding](https://www.microsoft.com/en-us/research/uploads/prod/2019/11/p894-li.pdf) | KDD | ``     | [PyTorch(Author)]()   |  ``  |
| [Transformer-Based Capsule Network For Stock Movements Prediction](https://aclanthology.org/W19-5511.pdf) | IJCAI-FinNLP | ``     | [PyTorch(Author)]()   |  ``  |
| [Knowledge-Driven Stock Trend Prediction and Explanation via Temporal Convolutional Network](https://aura.abdn.ac.uk/bitstream/handle/2164/12473/p678_deng.pdf;jsessionid=607F251CC0FA36201363178781DE3005?sequence=1) | WWW | `OD`     | [PyTorch(Author)]()   |  ``  |
| [Enhancing Stock Movement Prediction with Adversarial Training](https://www.ijcai.org/proceedings/2019/0810.pdf) | IJCAI | ``     | [TensorFlow(Author)](https://github.com/fulifeng/Adv-ALSTM)   |  `85`  |
| [CLVSA: A Convolutional LSTM Based Variational Sequence-to-Sequence Model with Attention for Predicting Trends of Financial Markets](https://www.ijcai.org/proceedings/2019/0514.pdf) | IJCAI | ``     | [PyTorch(Author)]()   |  ``  |
| [What You Say and How You Say It Matters: Predicting Stock Volatility Using Verbal and Vocal Cues](https://aclanthology.org/P19-1038.pdf) | ACL | `MM`     | [PyTorch(Author)](https://github.com/GeminiLn/EarningsCall_Dataset)   |  `52`  |
| [Exploring Graph Neural Networks for Stock Market Predictions with Rolling Window Analysis](https://arxiv.org/pdf/1909.10660.pdf) | NeurIPS-W | ``     | [PyTorch(Author)]()   |  ``  |
| [Temporal Relational Ranking for Stock Prediction](https://arxiv.org/pdf/1809.09441.pdf) | ACM-TIS | `OD`     | [TensorFlow(Author)](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking)   |  `176`  |
| [A Novel Approach to Short-Term Stock Price Movement Prediction using Transfer Learning](https://www.mdpi.com/2076-3417/9/22/4745/htm) | AS | `MS`     | [PyTorch(Author)]()   |  ``  |
| [DP-LSTM: Differential Privacy-inspired LSTM for Stock Prediction Using Financial News](https://arxiv.org/pdf/1912.10806.pdf) | NeurIPS | `OD`     | [PyTorch(Author)]()   |  ``  |
| []() | NeurIPS | ``     | [PyTorch(Author)]()   |  ``  |


### 2018

| Title    | Venue    | Type     | Code     | Star     |
|:-------- |:--------:|:--------:|:--------:|:--------:|
| [Stock Movement Prediction from Tweets and Historical Prices](https://aclanthology.org/P18-1183.pdf) | ACL | `OD`     | [PyTorch(Author)](https://github.com/yumoxu/stocknet-dataset)   |  `267`  |
| [Temporal Attention-Augmented Bilinear Network for Financial Time-Series Data Analysis](https://arxiv.org/pdf/1712.00975.pdf) | TNNLS | ``     | [PyTorch(Author)](https://github.com/viebboy/TABL)   |  `24`  |
| [Incorporating Corporation Relationship via Graph Convolutional Neural Networks for Stock Price Prediction](http://www.sdspeople.fudan.edu.cn/zywei/paper/chen-cikm2018.pdf) | CIKM | `MS`     | [PyTorch(Author)]()   |  ``  |
| [Hierarchical Complementary Attention Network for Predicting Stock Price Movements with News]() | CIKM | `OD`     | [PyTorch(Author)]()   |  ``  |
| [Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction](https://www.researchgate.net/publication/321604413_Listening_to_Chaotic_Whispers_A_Deep_Learning_Framework_for_News-oriented_Stock_Trend_Prediction) | WSDM | `OD`     | [PyTorch(Author)](https://github.com/donghyeonk/han)   |  `12`  |


### 2017

| Title    | Venue    | Type     | Code     | Star     |
|:-------- |:--------:|:--------:|:--------:|:--------:|
| [A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](https://arxiv.org/pdf/1704.02971.pdf) | IJCAI | ``     | [Keras(Author)](https://github.com/chensvm/A-Dual-Stage-Attention-Based-Recurrent-Neural-Network-for-Time-Series-Prediction)   |  `91`  |
| [Stock Price Prediction via Discovering Multi-Frequency Trading Patterns](https://www.eecs.ucf.edu/~gqi/publications/kdd2017_stock.pdf) | KDD | ``     | [Keras(Author)](https://github.com/z331565360/State-Frequency-Memory-stock-prediction)   |  `54`  |


### 2016

| Title    | Venue    | Type     | Code     | Star     |
|:-------- |:--------:|:--------:|:--------:|:--------:|
| [Knowledge-Driven Event Embedding for Stock Prediction](https://aclanthology.org/C16-1201.pdf) | COLING | ``     | [PyTorch(Author)]()   |  ``  |


### Previous Venues

| Title    | Venue    | Type     | Code     | Star     |
|:-------- |:--------:|:--------:|:--------:|:--------:|
| [Deep Learning for Event-Driven Stock Prediction](https://www.ijcai.org/Proceedings/15/Papers/329.pdf) | IJCAI | `OD`     | [PyTorch(Author)](https://github.com/vedic-partap/Event-Driven-Stock-Prediction-using-Deep-Learning)   |  `152`  |
| [Using Structured Events to Predict Stock Price Movement: An Empirical Investigation](https://emnlp2014.org/papers/pdf/EMNLP2014148.pdf) | EMNLP | `OD`     | [PyTorch(Author)]()   |  ``  |



### arXiv

| Title    | Date     | Type     | Code     | Star     |
|:-------- |:--------:|:--------:|:--------:|:--------:|
| [Multivariate Time Series Imputation by Graph Neural Networks](https://arxiv.org/pdf/2108.00298.pdf) | 2021.09.23 | `MS`     | -   |  ``  |
| [Artificial intelligence prediction of stock prices using social media](https://arxiv.org/pdf/2101.08986.pdf) | 2021.01.22 | `OD`     | -   |  ``  |
| [Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport]() | 2021.01.25 | `MS`     | -   |  ``  |
| [Trade When Opportunity Comes: Price Movement Forecasting via Locality-Aware Attention and Adaptive Refined Labeling](https://arxiv.org/pdf/2107.11972.pdf) | 2021.01.26 | ``     | -   |  ``  |
| [Stock price prediction using BERT and GAN](https://arxiv.org/pdf/2107.09055.pdf) | 2021.01.28 | ``     | -   |  ``  |
| [HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction](https://arxiv.org/pdf/1908.07999.pdf) | 2019.11.12 | ``     | -   |  ``  |
| [Online Multi-Agent Forecasting with Interpretable Collaborative Graph Neural Networks](https://arxiv.org/pdf/2107.00894.pdf) | 2021.07.02 | `Other`     | -   |  ``  |
| [Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx](https://arxiv.org/pdf/2104.05522.pdf) | 2021.04.23 | `Other`     | -   |  ``  |
| []() | 2021.01.28 | ``     | -   |  ``  |
| []() | 2021.01.28 | ``     | -   |  ``  |
| []() | 2021.01.28 | ``     | -   |  ``  |


## Awesome Repositories
- [deep-finance: Deep Learning for Finance](https://github.com/sangyx/deep-finance)
- [DL4Stock](https://github.com/jwwthu/DL4Stock)
- [Fintech Literature](https://github.com/ai-gamer/fintech-literature)
- [Deep learning models for traffic prediction](https://github.com/aptx1231/Traffic-Prediction-Open-Code-Summary)
- [Fintech-Survey: Paper & Code & Dataset Collection](https://github.com/NickHan-cs/Fintech-Survey)
- [FinNLP: Research sources for NLP in Finance](https://github.com/JLeeAI/FinNLP)
- [Graph-models-in-finance-application](https://github.com/jackieD14/Graph-models-in-finance-application)
- [GNN-finance: Curated list of Graph Neural Network Applications in Business and Finance](https://github.com/kyawlin/GNN-finance)
- [Deep Learning Time Series Forecasting](https://github.com/Alro10/deep-learning-time-series)


## Awesome Surveys
- [Fusion in stock market prediction: A decade survey on the necessity, recent developments, and potential future directions](https://reader.elsevier.com/reader/sd/pii/S1566253520303481?token=C34A4FE9A3C52B54CA7A0B3D3EE29993944747A7C4AB5B82E75A9F994F84BB09B6582CE406BB0C72EE2173000EBE59B7&originRegion=us-east-1&originCreation=20211102032356), 2021 Information Fusion
- [Applications of deep learning in stock market prediction: Recent progress](https://www.sciencedirect.com/science/article/abs/pii/S0957417421009441), 2021 ESA
- [Financial time series forecasting with deep learning : A systematic literature review: 2005–2019](https://arxiv.org/pdf/1911.13288.pdf), 2019 ASC
- [Natural language based financial forecasting: a survey](https://dspace.mit.edu/bitstream/handle/1721.1/116314/10462_2017_9588_ReferencePDF.pdf?sequence=2&isAllowed=y), 2017



## Awesome Blogs
- [Stock predictions with state-of-the-art Transformer and Time Embeddings](https://towardsdatascience.com/stock-predictions-with-state-of-the-art-transformer-and-time-embeddings-3a4485237de6)
- [当深度学习遇上量化交易——因子挖掘篇](https://www.linkresearcher.com/theses/d27a60fc-3174-42a9-a1ed-1c05d44b3327)


## Awesome Datasets
- [Top 5 Stock Market Datasets for Machine Learning](https://www.kaggle.com/learn-forum/167685)
- [Top 10 Stock Market Datasets for Machine Learning](https://imerit.net/blog/top-10-stock-market-datasets-for-machine-learning-all-pbm/)


## Awesome Code
- [Stock Prediction with Transformer (TensorFlow)](https://github.com/Stepka/Stock-Prediction-usning-Transformer-NN/blob/master/Stock_Prediction_usning_Transformer_NN.ipynb)
- [Transformer Time Series Prediction](https://github.com/oliverguhr/transformer-time-series-prediction)
- [DeepSeries: Deep Learning Models for time series prediction](https://github.com/EvilPsyCHo/Deep-Time-Series-Prediction)
