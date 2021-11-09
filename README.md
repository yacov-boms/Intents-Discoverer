# Intents-Discoverer
**Identifying Customer Intent in a Query**   
The program receives a user's free text query and identifies the intent by submitting it to a trained NN which returns a list of class probabilities.
This is a multiclass classification in which each intent is represented by a class. The intent of the best probability is chosen as the prediction.   
The network is built with **Tensorflow/Keras** and uses **Spacy** Word2Vev matrix of the trained set vocabulary as embedding layer. This is the only source of the net. It does not use an already trained model as transfer learning like transformer or BERT.   
The idea is that Spacy's word2ved contains word semantics, and therefore using it in an embedding layer helps the net's prediction.   

An example of the data set:   
![image](https://user-images.githubusercontent.com/54791267/140789908-68c00817-bd92-49a9-ac87-2e100d3934d5.png)

