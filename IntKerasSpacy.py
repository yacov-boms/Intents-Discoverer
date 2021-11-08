# A multi class text classification model
#
from IntClean import clean_example
import numpy as np 
import pandas as pd
import sys
sys.path.append("C:/Users/kobi/anaconda3/envs/tf/Lib/site-packages")
import tensorflow as tf # tf.__version__
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import time
import spacy        # v. 2.2.3
nlp = spacy.load("en_core_web_lg", disable=['ner', 'tagger', 'parser'])


df = pd.read_excel (r'C:\Users\Administrator\Desktop\Intents1.xlsx', sheet_name='Sheet1') 
examples = df['examples'].to_numpy()
labels = df['label'].to_numpy()
X = [clean_example(ex.lower()) for ex in examples]  # List of queries
y = labels
classes = list(set(df['label']))

# Creating The tokens Matrix (queries, max_query_length)
vocab_size = 1000
max_query_len = 15
oov_token = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(X)
padded_sequences = pad_sequences(sequences, 
                                 truncating='pre', 
                                 maxlen=max_query_len)
# Initial Embedding
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_matrix[i] = nlp(word).vector
   
# Model Construction
model = Sequential()
model.add(Embedding(input_dim=embedding_matrix.shape[0],
                    output_dim=embedding_matrix.shape[1],
                    weights=[embedding_matrix],
                    input_length=max_query_len,
                    trainable=False,
                    mask_zero=False))
model.add(GlobalAveragePooling1D())
model.add(Dense(300, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=Adam(learning_rate=0.01), 
              metrics=['accuracy'])
model.summary()

# for layer in model.layers: 
#     print(layer.get_config(), layer.get_weights())
model.save_weights('model.h5')
   
# Model Training
startTime = time.time()
epochs = 25
label_pred = []
for i in range(len(X)):
    X_train = np.delete(padded_sequences, i , axis=0)
    y_train = np.delete(y, i)
    model.load_weights('model.h5')
    history = model.fit(X_train, np.array(y_train), 
                        batch_size=8, epochs=epochs,verbose=0)
    result = model.predict(pad_sequences(tokenizer.texts_to_sequences([X[i]]), 
                                         truncating='pre', 
                                         maxlen=max_query_len))
    label = np.argmax(result)
    label_pred.append(label)
    print('query: ', i)
accuracy=sum(np.array(y)==np.array(label_pred))/len(y)
print('Accuracy: ', accuracy)
endTime = time.time()
print('Time:', endTime - startTime, 'seconds')
# Time: 1210.40
# n=177 ep=90 lr=0.001   dim=300   hidden=1  0.7231   
# n=177 ep=90 lr=0.001   dim=300   hidden=2  0.7118   
# n=177 ep=80 lr=0.001   dim=300   hidden=1  0.7231   
# n=177 ep=70 lr=0.001   dim=300   hidden=1  0.7457   
# n=177 ep=60 lr=0.001   dim=300   hidden=1  0.7288   
# n=177 ep=50 lr=0.001   dim=300   hidden=1  0.7231   
# n=177 ep=40 lr=0.001   dim=300   hidden=1  0.7231   0.7401  
# n=177 ep=30 lr=0.001   dim=300   hidden=1  0.7175   
# n=177 ep=30 lr=0.002   dim=300   hidden=1  0.7344   0.7175  0.7288  0.7344 
# n=177 ep=30 lr=0.003   dim=300   hidden=1  0.7175   
# n=177 ep=30 lr=0.004   dim=300   hidden=1  0.7288   
# n=177 ep=30 lr=0.005   dim=300   hidden=1  0.7231   
# n=177 ep=30 lr=0.006   dim=300   hidden=1  0.7344   
# n=177 ep=30 lr=0.007   dim=300   hidden=1  0.7231   
# n=177 ep=30 lr=0.008   dim=300   hidden=1  0.7401   
# n=177 ep=30 lr=0.009   dim=300   hidden=1  0.7683   0.7344  0.7288  
# n=177 ep=30 lr=0.01    dim=300   hidden=1  0.7231   0.7514  0.7683   
# n=177 ep=25 lr=0.01    dim=300   hidden=1  0.7627   0.7457  0.7514  0.7627
# n=177 ep=30 lr=0.011   dim=300   hidden=1  0.7514   
# n=177 ep=30 lr=0.012   dim=300   hidden=1  0.7344   
# n=177 ep=30 lr=0.012   dim=300   hidden=1  0.7175   
# n=177 ep=40 lr=0.001   dim=300   hidden=0  0.4293   

 
# Export to Excel  ------------------------------------------------------------
# writer = pd.ExcelWriter(r'C:\Users\Administrator\Desktop\IntentsKeras.xlsx', engine='xlsxwriter')
# df.to_excel(writer, sheet_name='Sheet1', index = False)
# workbook  = writer.book
# worksheet = writer.sheets['Sheet1']
# worksheet.set_column('A:A', 4, None)
# worksheet.set_column('B:B', 30, None)
# worksheet.set_column('C:C', 10, None)
# worksheet.set_column('D:D', 10, None)
# worksheet.set_column('E:E', 20, None)
# writer.save()

# token_list = ' '.join(X).split()
# token_list = [w.lower() for w in token_list]
# import nltk
# fdist = nltk.FreqDist(token_list)
# fdist.most_common()
