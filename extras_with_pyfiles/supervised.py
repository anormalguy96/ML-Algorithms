import math
import random

### --------------------
### 1. LINEAR REGRESSION
### --------------------

class LinearRegression:
    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for _ in range(self.epochs):
            # Stochastic Gradient Descent (update weights per sample)
            for i in range(n_samples):
                # Calculate y_pred for the current sample
                y_pred_i = sum(self.w[j] * X[i][j] for j in range(n_features)) + self.b
                error_i = y_pred_i - y[i]

                # Update weights
                for j in range(n_features):
                    self.w[j] -= self.lr * error_i * X[i][j]
                # Update bias
                self.b -= self.lr * error_i

    def predict(self, x):
        # Calculate prediction for a single sample
        return sum(wi * xi for wi, xi in zip(self.w, x)) + self.b

### -----------------------
### 2. LOGISTIC REGRESSION
### -----------------------

def sigmoid(z):
    # Handles potential overflow/underflow for very large/small z by clipping
    if z < -700: # Approx. log(1/float_max)
        return 0.0
    if z > 700:  # Approx. -log(float_min)
        return 1.0
    return 1 / (1 + math.exp(-z))

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        for _ in range(self.epochs):
            # Stochastic Gradient Descent (update weights per sample)
            for i in range(n_samples):
                # Calculate z for the current sample
                z_i = sum(self.w[j] * X[i][j] for j in range(n_features)) + self.b
                pred_i = sigmoid(z_i)
                error_i = pred_i - y[i]

                # Update weights
                for j in range(n_features):
                    self.w[j] -= self.lr * error_i * X[i][j]
                # Update bias
                self.b -= self.lr * error_i

    def predict(self, x):
        # Calculate z for a single sample
        z = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return int(sigmoid(z) >= 0.5)


### --------------------------
### 3. K-NEAREST NEIGHBOURS
### --------------------------

class K_Nearest_Neighbours:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean(self, a, b):
        # Optimized sum for squared differences using zip and generator expression
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def predict(self, x):
        # Calculate distances and store with labels
        distances = []
        for i in range(len(self.X_train)):
            dist = self._euclidean(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))

        # Sort distances
        distances.sort(key=lambda tup: tup[0])

        # Get top k labels
        top_k_labels = [label for _, label in distances[:self.k]]

        # Find the most common label among the top k
        if not top_k_labels:
            # Handle case where top_k is empty (e.g., if X_train is empty)
            return None # Or raise an error, depending on desired behavior

        # Manual counting for majority vote
        counts = {}
        for label in top_k_labels:
            counts[label] = counts.get(label, 0) + 1

        # Find label with max count
        max_count = -1
        majority_label = None
        for label, count in counts.items():
            if count > max_count:
                max_count = count
                majority_label = label
        return majority_label

    def predict_batch(self, X):
        return [self.predict(x_i) for x_i in X]


### -------------------
### 4. DECISION TREE
### -------------------

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def _gini(self, y):
        if not y:
            return 0.0
        # Manual counting for classes
        counts = {}
        for val in y:
            counts[val] = counts.get(val, 0) + 1

        impurity = 1.0
        n = len(y)
        for count in counts.values():
            impurity -= (count / n) ** 2
        return impurity

    def _split(self, X, y, feature_index, threshold):
        left_X, right_X, left_y, right_y = [], [], [], []
        for xi, yi in zip(X, y):
            if xi[feature_index] < threshold:
                left_X.append(xi)
                left_y.append(yi)
            else:
                right_X.append(xi)
                right_y.append(yi)
        return left_X, right_X, left_y, right_y

    def _best_split(self, X, y):
        n_samples = len(X)
        if n_samples == 0:
            return None, None, 0.0 # No split possible

        n_features = len(X[0])
        best_index, best_thresh, best_gain = None, None, 0.0
        base_gini = self._gini(y)

        for i in range(n_features):
            # Collect unique thresholds efficiently
            thresholds = set()
            for x_row in X:
                thresholds.add(x_row[i])

            for t in thresholds:
                left_X, right_X, left_y, right_y = self._split(X, y, i, t)

                if not left_y or not right_y: # Skip empty splits
                    continue

                # Calculate weighted Gini impurity
                gini = (len(left_y) / n_samples) * self._gini(left_y) + \
                       (len(right_y) / n_samples) * self._gini(right_y)
                gain = base_gini - gini

                if gain > best_gain:
                    best_index, best_thresh, best_gain = i, t, gain
        return best_index, best_thresh

    def _get_majority_class(self, y):
        if not y:
            return None # Or handle as an error/default
        counts = {}
        for val in y:
            counts[val] = counts.get(val, 0) + 1
        
        max_count = -1
        majority_class = None
        for val, count in counts.items():
            if count > max_count:
                max_count = count
                majority_class = val
        return majority_class

    def _build(self, X, y, depth):
        # Base cases for stopping tree growth
        if depth >= self.max_depth or not y or len(set(y)) == 1:
            return self._get_majority_class(y)

        index, threshold = self._best_split(X, y)

        if index is None: # No good split found, make it a leaf
            return self._get_majority_class(y)

        left_X, right_X, left_y, right_y = self._split(X, y, index, threshold)

        return {
            'index': index,
            'threshold': threshold,
            'left': self._build(left_X, left_y, depth + 1),
            'right': self._build(right_X, right_y, depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)

    def _predict(self, x, node):
        if not isinstance(node, dict): # Leaf node
            return node
        if x[node['index']] < node['threshold']:
            return self._predict(x, node['left'])
        else:
            return self._predict(x, node['right'])

    def predict(self, x):
        return self._predict(x, self.tree)


### -------------------
### 5. RANDOM FOREST
### -------------------

class RandomForest:
    def __init__(self, n_trees=5, max_depth=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def _bootstrap(self, X, y):
        n_samples = len(X)
        X_sample, y_sample = [], []
        for _ in range(n_samples):
            idx = random.randint(0, n_samples - 1)
            X_sample.append(X[idx])
            y_sample.append(y[idx])
        return X_sample, y_sample

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, x):
        votes = [tree.predict(x) for tree in self.trees]
        if not votes:
            return None # Handle case with no trees/predictions

        # Manual counting for majority vote
        counts = {}
        for vote in votes:
            counts[vote] = counts.get(vote, 0) + 1
        
        max_count = -1
        majority_vote = None
        for vote, count in counts.items():
            if count > max_count:
                max_count = count
                majority_vote = vote
        return majority_vote


### ------------------
### 6. SUPPORT VECTOR MACHINE (HARD MARGIN)
### ------------------

class SVM:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.w = [0.0] * n_features
        self.b = 0.0

        # Convert 0/1 labels to -1/1
        y_scaled = [1 if label == 1 else -1 for label in y]

        for _ in range(self.epochs):
            for i in range(n_samples):
                # Calculate the raw output
                raw_output = sum(self.w[j] * X[i][j] for j in range(n_features)) + self.b

                condition = y_scaled[i] * raw_output >= 1

                if condition:
                    # Update weights (only regularization term)
                    for j in range(n_features):
                        self.w[j] -= self.lr * (2 * self.w[j])
                else:
                    # Update weights (regularization + hinge loss gradient)
                    for j in range(n_features):
                        self.w[j] -= self.lr * (2 * self.w[j] - y_scaled[i] * X[i][j])
                    # Update bias
                    self.b += self.lr * y_scaled[i]

    def predict(self, x):
        raw = sum(wj * xj for wj, xj in zip(self.w, x)) + self.b
        return 1 if raw >= 0 else 0