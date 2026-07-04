# Sentiment Analysis Engine

## Project Overview
The Sentiment Analysis Engine is a computational tool designed to process and analyze textual data to determine its underlying emotional tone. Built using Python, the system leverages the TextBlob library to accurately classify text into positive or negative sentiments based on a trained Naive Bayes classifier. This project transforms unstructured text into structured, actionable insights using established Natural Language Processing (NLP) techniques.

## Core Features
* Lexicon-Based Analysis: Rapidly evaluates input text to calculate probability distributions, delivering immediate and quantifiable sentiment scores.
* Command-Line Interface (CLI): Built-in CLI for seamless testing of custom text inputs directly from the terminal.
* Efficient Data Handling: Capable of processing varying lengths of textual data, utilizing a pre-trained dataset of varied examples.
* Structured Output: Provides clear numerical confidence scores and distinct sentiment categories (POSITIVE or NEGATIVE).
* Object-Oriented Design: Modular and scalable architecture that allows for easy integration into larger applications or data pipelines.

## Technical Architecture
* Programming Language: Python
* Core NLP Framework: TextBlob (built on NLTK)
* Algorithm: Naive Bayes Classifier
* Key Libraries: textblob, argparse, typing

## Installation and Setup
To run this project locally, ensure you have Python installed, then install the required dependencies.

1. Install the TextBlob library:
   `pip install textblob`

2. Download the required NLTK corpora:
   `python -m textblob.download_corpora`

## Usage Instructions

### Default Execution
Run the script without any arguments to train the model, evaluate it against test data, and classify a default set of sentences.

`python main.py`

### Custom Text Analysis via CLI
Use the `--text` argument to analyze a specific sentence directly from the command line. Ensure the sentence is enclosed in quotation marks.

`python main.py --text "The new update completely broke my application and I am frustrated."`

## Potential Applications
* Customer Experience Monitoring: Automatically categorizing product reviews to identify areas for service improvement.
* Social Media Listening: Tracking brand perception and public opinion across various digital platforms.
* Market Research: Analyzing survey responses and open-ended feedback to extract quantitative metrics from qualitative data.
