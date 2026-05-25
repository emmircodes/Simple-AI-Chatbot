import random #for choosing random response
import json #to access json file yes
import pickle #for serialization(convert object state into format that can be stored)
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer #to make word with verbs (eg.works,worked) as a same word

lemmatizer = WordNetLemmatizer()

#---------LOAD TRAINING DATA----------

intents = json.loads(open('intents.json').read()) #creating intents object(or clled dictionary) by readin json file as text and passin in to loads functions.

#lists
words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

#iterate the intent (ulang intent)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) #tokenize = to individualize the word in a sentence(split "i am ady" to "i", "am", "ady")
        words.extend(word_list) #taking the word_list content and append it to the list.
        documents.append((word_list, intent['tag'])) #taking the list and appending it to the list
        if intent['tag'] not in classes:
            classes.append(intent['tag']) #to check if the intent is in classes list or not

#print(documents)

#----------PREPARE TRAINING DATA----------

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

#print(words)

classes = sorted(set(classes))

#save them into files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#characters n words are not numerical values, neural network need numerical values.
# we need to represents this words into numerical value.
# we are going to set individual words into value 0 or 1.

training = []
output_empty = [0] * len(classes) #templates of 0 and * length of classes cuz we need as many zero as there are

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0) #if the word exist in the documents then append 1, if no then 0

    #for the output
    output_row = list(output_empty) #copy output_empty list
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row) #append bag n output_row to training list
    #we goin to use this training list to train the neural network

#final step pre-processing before buildin the neural network
random.shuffle(training) #shuffle the training data
training = np.array(training) #turn it into numpy array

#split the data into x and y values, : means all data in the training list.
#this will be the label that we will be goin to use in neural network
train_x = list(training[:, :len(words)])
train_y = list(training[:, len(words):])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.keras', hist)
print('done')