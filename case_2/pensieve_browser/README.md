# Browser Related Expriment

Original Pensieve implement RL Algorithm in remote computation server in order to avoid possible 
huge computation burden of a neural network. In this expriment, we run the whole pensieve system
entirely in the user client with the help of `tensorflow.js`.

Pensieve modified `dash.js` and added several ABR algorithm including the RL based one. We start from
code in Pensieve official github repository and add four abr algorithm:

* Pensieve(Entirely in browser, model is exported from the pre-trained one provided by authors of pensieve)
* Simplified version of Pensieve with LIME
* Simplified version of Pensieve with LEMNA
* Simplified version of Pensieve with our algorithm

# Reproduce our expriment

TBA
