from sklearn import tree

APPLE = 0
ORANGE = 1
BUMPY = 2
SMOOTH = 3

#Features come in the form [mass,texture]
features = [[200,BUMPY],[210,BUMPY],[150,SMOOTH],[160,SMOOTH],[190,BUMPY]]
labels = [ORANGE,ORANGE,APPLE,APPLE,ORANGE]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)

print(clf.predict([[200,BUMPY],[150,SMOOTH]]))