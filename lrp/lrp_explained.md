# LRP 
For the explanation of the implemented LRP, we will mainly use chapter 10 [Layer-Wise Relevance Propagation: An Overview](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10) of the paper [Explainable AI: Interpreting, Explaining and Visualizing Deep Learning](https://link.springer.com/book/10.1007/978-3-030-28954-6).

## quick overview
LRP operates by propagating the prediction f(x) backward in the neural network, by means of purposely designed local propagation rules.

## LRP implementation
The LRP package consists of the following four files. 

### __init__.py
Files named __init__.py are used to mark directories on disk as Python package directories. This is done so that lrp can be called in the main Jupyter notebook.

### lrp.py
This is the main implementation of the lrp. It happens in multiple steps: 
* 1
* 2
* 3
* 4

### lrp_layers.py
Here can be found the layers for layer-wise relevance propagation.


### lrp_filter
Implements filter method for relevance scores.

