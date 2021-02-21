from PIL import Image
import pytesseract
import os
import cv2
from gtts import gTTS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# contacting os
if not os.path.exists('content_frames'):
    os.makedirs('content_frames')
test_vid = cv2.VideoCapture('1612947719607.mp4')
# converting video into frames
index = 0
while test_vid.isOpened():
    ret, frame = test_vid.read()
    if not ret:
        break
    name = './content_frames/frame' + str(index) + '.png'
    print('Extracting frames...' + name)
    cv2.imwrite(name, frame)
    index = index + 1
    if cv2.waitKey(0) == 27:
        break
test_vid.release()
cv2.destroyAllWindows()

# convert and display image into words
demo = Image.open("./content_frames/frame1.png")
text = pytesseract.image_to_string(demo, lang = 'eng')
print(text)

# convert test into audio
mytext = text
language = 'en'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("DOC.mp3")

# predict the emotion using audio
sentence = text
sid_obj = SentimentIntensityAnalyzer()
sentiment_dict = sid_obj.polarity_scores(sentence)
print("Overall sentiment dictionary is : ", sentiment_dict)
print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
print("Sentence Overall Rated As", end = " ")
if sentiment_dict['compound'] >= 0.05:
    print("Positive")
elif sentiment_dict['compound'] <= - 0.05:
    print("Negative")
else:
    print("Neutral")

# converting audio into pie chart
labels = ['negative', 'neutral', 'positive']
sizes =sentiment_dict['neg'], sentiment_dict['neu'], sentiment_dict['pos']
colors =['red','black','green']
plt.pie(sizes,labels=labels, colors=colors,startangle=140)
plt.legend(labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()





