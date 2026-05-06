import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_cifar10_batch(file): # load Data
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    X = dict[b'data']
    Y = dict[b'labels']
    return X, Y

def load_cifar10(data_dir): # load DataSet
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(data_dir, 'data_batch_%d' % (b, ))
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))
    return Xtr, Ytr, Xte, np.array(Yte)

class KNearestNeighbor: # 
    def __init__(self):
        pass

    def train(self, X, y):
        # remembers all the training data
        self.X_train = X
        self.y_train = y

     # L1 distance 
    def compute_distances_l1(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.sum(np.abs(self.X_train - X[i, :]), axis=1)
        return dists

    # L2 distance 
    def compute_distances_l2(self, X):
        # L2 distance squared: (X - Y)^2 = X^2 + Y^2 - 2XY
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        
        # (X - Y)^2 = X^2 - 2XY + Y^2
        X_squared = np.sum(X**2, axis=1, keepdims=True)
        Y_squared = np.sum(self.X_train**2, axis=1)
        XY = np.dot(X, self.X_train.T)
        
        dists = np.sqrt(X_squared - 2*XY + Y_squared)
        return dists

    #Finding K Nearest Neighbors
    def predict(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]
            y_pred[i] = Counter(closest_y).most_common(1)[0][0]
        return y_pred


def main():
    print("Loading CIFAR-10 dataset")
    cifar10_dir = 'cifar-10-batches-py'
    try:
        X_train, y_train, X_test, y_test = load_cifar10(cifar10_dir)
    except Exception as e:
        print(f"Error loading CIFAR-10 from {cifar10_dir}: {e}")
        return

    # Subsample data
    num_training = 5000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 1000
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    # Convert to float to avoid overflow
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    classifier = KNearestNeighbor()

    # Setting Parameter
    k_choices = [1, 3, 5, 7, 9]
    metrics = ['L1', 'L2']
    num_folds = 5

    print(f"Performing {num_folds}-fold Cross-Validation")
    
    # K-Cross Validation : Split training data into folds 
    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)

    k_to_accuracies = {}
    
    for metric in metrics:
        print(f"Evaluating {metric} metric via cross-validation:")
        for k in k_choices:
            fold_accuracies = []
            for i in range(num_folds):
                # We use fold i as validation set, and others as training set
                X_val_fold = X_train_folds[i]
                y_val_fold = y_train_folds[i]
                
                X_train_fold = np.concatenate([X_train_folds[j] for j in range(num_folds) if j != i])
                y_train_fold = np.concatenate([y_train_folds[j] for j in range(num_folds) if j != i])
                
                classifier.train(X_train_fold, y_train_fold)
                
                if metric == 'L1':
                    dists = classifier.compute_distances_l1(X_val_fold)
                else:
                    dists = classifier.compute_distances_l2(X_val_fold)
                    
                y_val_pred = classifier.predict(dists, k=k)
                num_correct = np.sum(y_val_pred == y_val_fold)
                accuracy = float(num_correct) / len(y_val_fold)
                fold_accuracies.append(accuracy)
                
            avg_acc = np.mean(fold_accuracies)
            k_to_accuracies[(metric, k)] = avg_acc
            print(f"Metric: {metric}, k: {k}, Cross-validation Accuracy: {avg_acc:.3f}")


    # ----------------- TEST ---------------------


    best_l1_acc = -1
    best_l1_k = -1
    best_l2_acc = -1
    best_l2_k = -1

    for metric, k in k_to_accuracies:
        acc = k_to_accuracies[(metric, k)]
        if metric == 'L1' and acc > best_l1_acc:
            best_l1_acc = acc
            best_l1_k = k
        elif metric == 'L2' and acc > best_l2_acc:
            best_l2_acc = acc
            best_l2_k = k

    print(f"Best Cross-Validation L1 Accuracy: {best_l1_acc:.3f} with k={best_l1_k}")
    print(f"Best Cross-Validation L2 Accuracy: {best_l2_acc:.3f} with k={best_l2_k}")

    # Now evaluate on the actual test set with the best Ks
    print("Evaluating the best models on the TEST set")
    classifier.train(X_train, y_train)

    # Test best L1
    print("Computing L1 distances for TEST set")
    dists_l1 = classifier.compute_distances_l1(X_test)
    best_l1_y_pred = classifier.predict(dists_l1, k=best_l1_k)
    test_l1_acc = float(np.sum(best_l1_y_pred == y_test)) / num_test
    print(f"Test L1 Accuracy (k={best_l1_k}): {test_l1_acc:.3f}")

    # Test best L2
    print("Computing L2 distances for TEST set")
    dists_l2 = classifier.compute_distances_l2(X_test)
    best_l2_y_pred = classifier.predict(dists_l2, k=best_l2_k)
    test_l2_acc = float(np.sum(best_l2_y_pred == y_test)) / num_test
    print(f"Test L2 Accuracy (k={best_l2_k}): {test_l2_acc:.3f}")

    # Confusion matrix
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Save L1 confusion matrix
    cm_l1 = confusion_matrix(y_test, best_l1_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_l1, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (L1, k={best_l1_k})')
    plt.savefig('knn_confusion_matrix_l1.png')
    
    # Save L2 confusion matrix
    cm_l2 = confusion_matrix(y_test, best_l2_y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_l2, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (L2, k={best_l2_k})')
    plt.savefig('knn_confusion_matrix_l2.png')

if __name__ == '__main__':
    main()
