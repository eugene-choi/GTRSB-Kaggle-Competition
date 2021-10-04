# GTRSB-Kaggle-Competition
Project for CSCI-GA-2271: Computer Vision, taught by Prof. Rob Fergus, Fall 2021.

# Performance:
This implementation achieved the first place out of a total of 137 participants in both public and private leaderboard from the Computer Vision class. It achieved 0.99730 accuracy on the public leaderbaord and an even higher score of 0.99762 on the private leaderboard.

## Motivation
Sarcasm detection is a challenge for sentiment analysis due to its difference between surface meaning and intended meaning. In this pa- per, we test state-of-the-art models (BERT, RoBERTa) on their grasp on sarcasm using sentiment analysis. We append sarcastic sen- tences from Sarcasm Corpus V2 and SemEval- 2018 to the IMDB and SST2 dataset then ob- serve behavioral changes. We expect predicted labels to become negative or stay negative as sarcasm has an inherent negative sentiment. Our results show that without additional train- ing, BERT-style models lack the knowledge to process sarcasm and its negative sentiment, and manipulating the length and position of the sarcasm concatenation did not show ob- vious trends. Different types of sarcasm and datasets also do not yield notable differences.

## Data Collection
We categorize our datasets into two subcategories: sentiment and sarcasm. Sentiment datasets include IMDB Review Dataset and Stanford Sentiment Treebank 2, while the sarcasm datasets include Sarcasm Corpus V2 and SemEval-2018.

## Modeling and Analysis
Using the BERT-style models, we perform senti- ment analysis on the IMDB Review Dataset and SST2 datasets. First, we get baselines results from the sentiment datasets to ensure the model is work- ing as expected. Then, we perform sentiment analysis on the subset containing only sarcastic data in the Sarcasm Corpus V2 and SemEval-2018 datasets. We change all labels from 1 (marked for sarcasm) to 0 (negative) to fit the task. Then, we collect examples of what the models identify as negative sentiment, then append them to review datasets as perturbations. Perturbation cases are created by length (x < 128 or 128 ≤ x < 256, x = characters) and position (appending at the begin- ning or end of the review text). We designing our tests by referring to the Checklist framework. If models can successfully detect negative sentiment in sarcasm, we expect cases attached to positive reviews to change labels to negative, similar to a directional expectation test, and cases attached to negative reviews to retain the same label of neg- ative, similar to an invariance test. While these combinations may not appear in real life, it can serve as a simulation for when sarcasm is embed- ded in longer contexts.

## Spatial Transformer Network

## Stochastic Weight Averaging

## Paper
Deep neural network for traffic sign recognition systems: An analysis of spatial transformers and stochastic optimisation methods (Arcos-Garcı’a et al., 2018) [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608018300054).
Spatial Transformer Networks (Jaderberg et al., 2016) [paper] (https://arxiv.org/abs/1506.02025).
Averaging Weights Leads to Wider Optima and Better Generalization (Izmailov et al., 2019) [paper] (https://arxiv.org/abs/1803.05407).

