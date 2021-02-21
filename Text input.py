from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
a=input()
sentence=a
sid_obj = SentimentIntensityAnalyzer()
sentiment_dict = sid_obj.polarity_scores(sentence)
print("Overall sentiment dictionary is : ", sentiment_dict)
print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
print("Sentence Overall Rated As", end = " ")
labels = ['negative', 'neutral', 'positive']
sizes = [sentiment_dict['neg'], sentiment_dict['neu'], sentiment_dict['pos']]
colors =['red','black','green']
plt.pie(sizes, labels=labels, colors=colors,startangle=140)
plt.legend( labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()

