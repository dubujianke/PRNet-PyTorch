from sklearn import datasets
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

# dot - Tpng tree.dot - o example.png

datads = pd.read_excel(r'D:\stock\Test\res\report\root\node1\zt_bak__.xlsx', index_col=None)
datads.fillna(1000, inplace=True)

y_train_ = datads.iloc[:, 4].map(lambda x: x > 7.3 and 1 or 0).values

x_train_ = datads.iloc[:, 5:].values
x_train, x_test, y_train, y_test = train_test_split(x_train_, y_train_, test_size=0.1)

a = 10
c = a == 0 and "zero" or "not_zero"
print(type(c), c) # <class 'str'> not_zero

print(x_train_.shape)
print(y_train_.shape)
tree_clf = DecisionTreeClassifier(max_depth=5, criterion='gini')
tree_clf.fit(x_train, y_train)

y_train_hat = tree_clf.predict(x_train)
print("acc score:", accuracy_score(y_train, y_train_hat))
print(tree_clf.feature_importances_)
#
#
dot_data = tree.export_graphviz(
    tree_clf,
    out_file='tree.dot',
    # feature_names=iris.feature_names[:],
    # class_names=iris.target_names,
    rounded=True,
    filled=True
)



