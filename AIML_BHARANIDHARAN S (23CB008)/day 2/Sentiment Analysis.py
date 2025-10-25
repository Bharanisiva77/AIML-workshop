import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = {
    "review":[
        "The Movie was Nice and I loved it",
        "Absolutely Terrible, waste of time",
        "I enjoyed every moment",
        "I hated this movie",
        "It was a materpiece of cine"
    ],
    "sentiment":[
        "positive","negative","positive","negative","positive"
    ]
}


df = pd.DataFrame(data)
vectorizer = CountVectorizer()
x=vectorizer.fit_transform(df['review']).toarray()
le=LabelEncoder()
y=le.fit_transform(df['sentiment'])

x_train,x_test,y_train,y_test=train_test_split(
x,y,test_size=0.2,random_state=42)

# Neural Networks Build
model=Sequential(
    [
        Dense(8,input_dim=x.shape[1],activation='relu'),
        Dense(4,activation='relu'),
        Dense(1,activation='sigmoid')
    ]
)

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=100,batch_size=2,verbose=0)

loss,acc=model.evaluate(x_test,y_test)
print("Model Accuracy: ",round(acc*100,2),"%")

sample_review = ["The Movie was Really Good"]
sample_vector = vectorizer.transform(sample_review).toarray()
prediction=model.predict(sample_vector)

print("Predicted Sentiment: ","Positive" if prediction[0][0]>0.5 else "Negative")