"""
Sentiment Analyzer using TextBlob's NaiveBayesClassifier
==========================================================

This project trains a Naive Bayes classifier to detect whether a piece of
text has POSITIVE or NEGATIVE sentiment.

HOW IT WORKS :
- Naive Bayes is a probabilistic classifier. It looks at the WORDS in a
  sentence and learns which words tend to appear in positive vs negative
  examples.
- "Naive" because it assumes each word is independent of the others
  (not strictly true, but works surprisingly well for text).
- TextBlob wraps this classic algorithm (from the `nltk` library) in a
  very simple, beginner-friendly API.

SETUP (run this once in your terminal):
    pip install textblob
    python -m textblob.download_corpora   # downloads NLTK data TextBlob needs

Then run this file with:
    python sentiment_analyzer.py
"""



from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob


# ---------------------------------------------------------------------------
# STEP 1: Training data
# ---------------------------------------------------------------------------
# Each tuple is (sentence, label). The more examples you add, and the more
# varied they are, the better the classifier will generalize.
train_data = [
    ("I love this movie, it was amazing!", "pos"),
    ("What a fantastic performance by the whole cast.", "pos"),
    ("This is the best day of my life.", "pos"),
    ("I'm so happy with how this turned out.", "pos"),
    ("The food at this restaurant was delicious.", "pos"),
    ("She did an excellent job on the project.", "pos"),
    ("I really enjoyed reading this book.", "pos"),
    ("Great service and a wonderful atmosphere.", "pos"),
    ("This phone works perfectly, I'm impressed.", "pos"),
    ("What a beautiful sunny day, I feel great.", "pos"),

    ("I hate this movie, it was terrible.", "neg"),
    ("This is the worst experience I've ever had.", "neg"),
    ("I'm so disappointed with the service.", "neg"),
    ("The food was cold and tasted awful.", "neg"),
    ("He did a horrible job on the project.", "neg"),
    ("I really regret buying this product.", "neg"),
    ("Terrible customer support, very frustrating.", "neg"),
    ("This phone is broken and useless.", "neg"),
    ("What a gloomy and depressing day.", "neg"),
    ("I feel awful about how this turned out.", "neg"),
]

# ---------------------------------------------------------------------------
# STEP 2: Test data (used to measure accuracy on UNSEEN examples)
# ---------------------------------------------------------------------------
test_data = [
    ("This is an amazing product, I love it!", "pos"),
    ("I'm thrilled with the results.", "pos"),
    ("Such a wonderful and pleasant surprise.", "pos"),

    ("This was a complete waste of money.", "neg"),
    ("I'm furious about how I was treated.", "neg"),
    ("The movie was boring and disappointing.", "neg"),
]


def main():
    # -----------------------------------------------------------------
    # STEP 3: Train the classifier
    # -----------------------------------------------------------------
    print("Training the Naive Bayes classifier...")
    classifier = NaiveBayesClassifier(train_data)
    print("Training complete!\n")

    # -----------------------------------------------------------------
    # STEP 4: Evaluate accuracy on the test set
    # -----------------------------------------------------------------
    accuracy = classifier.accuracy(test_data)
    print(f"Accuracy on test data: {accuracy * 100:.1f}%\n")

    # See which words the classifier found most useful
    print("Most informative features:")
    classifier.show_informative_features(5)
    print()

    # -----------------------------------------------------------------
    # STEP 5: Classify new, custom sentences
    # -----------------------------------------------------------------
    new_sentences = [
        "I can't believe how good this was, absolutely loved it!",
        "This is dreadful, I want a refund.",
        "The weather today is quite nice.",
        "My internet connection keeps dropping and it's driving me crazy.",
    ]

    print("Classifying new sentences:")
    print("-" * 60)
    for sentence in new_sentences:
        # Method 1: use the classifier directly
        label = classifier.classify(sentence)

        # Method 2: use TextBlob, which also gives you a probability
        blob = TextBlob(sentence, classifier=classifier)
        prob_dist = classifier.prob_classify(sentence)
        pos_prob = prob_dist.prob("pos")
        neg_prob = prob_dist.prob("neg")

        sentiment_word = "POSITIVE" if label == "pos" else "NEGATIVE"
        print(f"Text: {sentence}")
        print(f"  -> Predicted: {sentiment_word}")
        print(f"  -> Confidence: pos={pos_prob:.2f}, neg={neg_prob:.2f}")
        print()


if __name__ == "__main__":
    main()