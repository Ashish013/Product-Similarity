# Image-Similarity
This repository contains code for Product-Similarity, developed as a part of Hack-a-sol hackathon @IIIT Naya Raipur.

The model is trained on 1500 images of grocery products and employs as Siamsese style network model.
At the core of the sister networks sits a ResNet-50 architecture fine tuned on the 1500 images of groceries and trained against both Contrastive and Binary Cross Entropy loss, each individually.

The application lets the user toggle between the 2 models trained on each of the losses respectively. A quick summary of the 2 losses can be articulated as :

## Contrastive loss
Let the embeddings from the sister networks be **embA , embB**.


Then the contrastive loss can be defined as:

![](https://miro.medium.com/max/3478/1*g8TVcxgVigHtYEmYilsfQw.png)
where :

* Dw is the euclidean distance between the two embeddings **embA** and **embB**.
* m is the margin between 2 distinct classes.

## Binary Cross Entropy loss
The euclidean distance between the two embeddings from sister networks are calculated and are passed through a dense layer with a sigmoid activation which predicts the similarity between the 2 input images. This output is then backpropogated using the binary cross entropy loss to better predict the image similarity.

The Binary Cross Entropy loss can be defined as:

![](https://static.packt-cdn.com/products/9781789132212/graphics/a9c5e929-2307-45ee-ac68-59bf520354b4.png)
where:

* pi is the predicted class.
* yi is the actual class.
