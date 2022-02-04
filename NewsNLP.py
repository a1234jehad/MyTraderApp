from  urllib.request import urlopen, Request
import keras.utils.np_utils
import numpy as np
from  bs4 import BeautifulSoup
import pandas as pd
from  sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf
finviz_url = "https://finviz.com/quote.ashx?t="
ticker = "MSFT"
url = finviz_url+ticker
req = Request(url=url, headers={'user-agent' : 'my-app/0.0.1'})
response = urlopen(req)
print(response)
html = BeautifulSoup(response)
# print(html)

news_table = html.find(id='news-table')
# print(news_table)
dataRows = news_table.findAll('tr')
# print(dataRows)

df = pd.DataFrame(columns=['News_Title','Time'])

for i,table_row in enumerate(dataRows):
    a_text = table_row.a.text
    td_text = table_row.td.text
    df = df.append({"News_Title" : a_text, 'Time' : td_text}, ignore_index=True)
print(df)

dff = pd.read_csv("FinancialNews.csv", encoding='latin-1')

dff.columns =  ['Sentiment','SentimentText']
mapper = {
    'negative':0,
    'neutral' :1,
    'positive':2,
}
dff.Sentiment = dff.Sentiment.map(mapper)
print(dff)

train,valid = train_test_split(dff, test_size=0.2)
train_text = np.array(train['SentimentText'].tolist().copy())
labels = keras.utils.np_utils.to_categorical(train['Sentiment'].astype('int64'))

valid_text = np.array(valid['SentimentText'].tolist().copy())
labels_valid = keras.utils.np_utils.to_categorical(valid['Sentiment'].astype('int64'))

vocab_size = 1000
embedding_dim = 16
max_length = 142
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(train_text)

# tokenizer.word_index
sequences = tokenizer.texts_to_sequences(train_text)

padded = pad_sequences(sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)
testing_sequence = tokenizer.texts_to_sequences(valid_text)
testing_padded = pad_sequences(testing_sequence,maxlen=max_length,padding=padding_type,truncating=trunc_type)

with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)

model = keras.Sequential([
    keras.layers.Embedding(vocab_size,embedding_dim,input_length = max_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16,activation= 'relu'),
    keras.layers.Dense(3,activation= 'softmax')
])

model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
print(model.summary())

num_epochs = 30
history = model.fit(padded,labels,epochs=num_epochs,validation_data=(testing_padded,labels_valid))

phrase = df.News_Title

testing_sequence = tokenizer.texts_to_sequences(phrase)
testing_padded = pad_sequences(testing_sequence,maxlen=max_length,padding=padding_type,truncating=trunc_type)
pred = model.predict(testing_padded)
print(pred)
# classes = np.argmax(pred,axis=-1)
# dict_sentiment = {
#     0:'negative',
#     1:'neutral' ,
#     2:'positive',
# }

df['Sentiment'] = np.argmax(pred,axis=-1)
# print(df.head(50))
print(df.Sentiment.value_counts())
