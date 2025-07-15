# ML-Algorithms

I built this repository so you can see exactly how classic ML & DL models work—step by step—without hiding behind big frameworks.
Everything here is written in pure Python, with only `math`, `random` and `matplotlib` as helpers. No NumPy, no TensorFlow.

And no Scikit-learn - so you can watch it learn :D. 
Now it just throws me a ```ValueError: humour not defined```.

Anyway, I was joking.

## What’s Inside

- **supervised/**  
  Linear Regression, Logistic Regression, KNN, Decision Tree, Random Forest, SVM  
- **unsupervised/**  
  K‑Means, DBSCAN, PCA, Hierarchical Clustering  
- **reinforcement_learning/**  
  Q‑Learning, SARSA
- **deep_learning/**  
  Perceptron, MLP, CNN Basics, and a basic RNN cell  
- **extras_with_pyfiles/**  
  Most of the above algorithms refactored into neat `.py` modules for easy importing (I did not involve 3 of supervised algos, the PCA from unsupervised, and RNN & CNN from DL since it is just a demo `.py` file).
- **utils/**  
  Handy functions for metrics, plotting and toy data generation  

## Why This Repo Exists
Most tutorials jump straight to libraries, leaving you wondering what’s really happening under the hood. Here, you’ll:

- Read and run code that you wrote yourself, line by line  
- Learn the math and logic behind each algorithm  
- Tweak parameters, try different data, and see how it all changes  

It’s perfect for students, hobbyists, or anyone curious about how those “black‑box” models actually work.

## License
This project is MIT‑licensed — check out the LICENSE file for details.
