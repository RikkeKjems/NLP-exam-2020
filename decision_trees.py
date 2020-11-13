pip install -U scikit-learn

from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

clf.predict([[2., 2.]])
clf.predict_proba([[2., 2.]])


from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

tree.plot_tree(clf) 

# https://scikit-learn.org/stable/modules/tree.html