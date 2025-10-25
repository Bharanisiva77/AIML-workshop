import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

import re
import kagglehub

path = kagglehub.dataset_download("crowdflower/twitter-airline-sentiment")

print("Path to dataset files:", path)

df = pd.read_csv(f"{path}/Tweets.csv")


def clean_text(text):
  if not isinstance(text, str):
    return ""
  text=re.sub(r"https","",text)
  text=re.sub(r"#\S+","",text)
  text=re.sub(r"@\S+","",text)
  text=re.sub(r"[^a-zA-Z\s]","",text)
  text=text.lower()
  return text
df['clean_text']=df['text'].apply(clean_text)
vectorizer = CountVectorizer(max_features=5000)
x=vectorizer.fit_transform(df['clean_text']).toarray()

label_encoder= LabelEncoder()
y=label_encoder.fit_transform(df['airline_sentiment'])
y=to_categorical(y)

x_train,x_test,y_train,y_test, text_train, text_test = train_test_split(x,y,df['clean_text'],test_size=0.2,random_state=42)

model=Sequential()
model.add(Dense(120,input_dim=x.shape[1],activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(y.shape[1],activation='softmax'))

optimizer = SGD(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,batch_size=32,validation_data=(x_test,y_test))

loss, accuracy = model.evaluate(x_test,y_test)
print("Test Accuracy: ",round(accuracy*100,2),"%")

predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
predicted_sentiments = label_encoder.inverse_transform(predicted_classes)

print("\nSample Predictions:")
for i in range(30):
  print(f"Input: {text_test.iloc[i]} -> Predicted Sentiment: {predicted_sentiments[i]}")