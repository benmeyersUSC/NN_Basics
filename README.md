![NN_Basics](exampleVisualization.png)

This program initializes a random input vector (default size 16) and a weight matrix that is filled 
with 0s and is initialized to have dimensions Y.size rows by X.size columns. 

While the inputs and outputs do not represent any meaningful encoding, this program
illustrates the process by which a loss function is calculated, gradients are computed, 
and weights are tweaked to iteratively minimize loss. 

While a microcosm for the massive intertwined neural networks that operate within modern LLMs, 
it shows the intuition behind the algorithm (backpropagation/gradient descent) that allows us to
build networks that learn from data. 