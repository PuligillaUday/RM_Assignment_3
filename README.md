Sentiment Analysis on Yelp Reviews Using BERT
Project Overview
This project involves fine-tuning a BERT (Bidirectional Encoder Representations from Transformers) model to perform sentiment analysis on the Yelp Polarity dataset. The primary goal is to classify Yelp reviews as either positive or negative, providing insights into customer opinions. This project demonstrates the power of Large Language Models (LLMs) in understanding and processing natural language tasks.

Table of Contents
Project Overview
Dataset
Model
Installation
Usage
Results
Visualization
References
Contributors
Dataset
The dataset used in this project is the Yelp Polarity dataset, which is part of the Hugging Face Datasets library. The dataset consists of 560,000 training samples and 38,000 test samples, with an equal distribution of positive and negative sentiment labels.

Source: Hugging Face Datasets - Yelp Polarity

Model
The model used in this project is based on the bert-base-uncased architecture from the Hugging Face Transformers library. The model is fine-tuned using the Yelp Polarity dataset to perform binary classification (positive or negative sentiment).

Results
The fine-tuned BERT model achieves high accuracy in classifying Yelp reviews. Key evaluation metrics include:

Evaluation Loss: 0.3792
Precision: High precision in distinguishing between positive and negative sentiments.
Recall: Balanced recall for both sentiment classes.
AUC: High Area Under the Curve (AUC) score, indicating strong model performance.
Visualization
Several visualizations are provided to help understand the model's performance:

Training Loss vs. Epochs
Confusion Matrix
Precision-Recall Curve
ROC Curve
Distribution of Review Lengths

References
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT 2019.
Hugging Face Datasets Library. Hugging Face. Available at: https://huggingface.co/datasets [Accessed 15 August 2024].

