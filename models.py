import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree

# Llamamos a la clase del módulo de creamos previamente:
from utils import Utils


class Models:
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    def tree_training(self, X,y):
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 99)

        # Caracteristicas del arbol:
        max_depth = 7
        t = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth = max_depth)

        #entrenamos
        model = t.fit(x_train, y_train)

        #Evaluación
        score_train = model.score(x_train, y_train)
        score_test = model.score(x_test, y_test)

        return score_test, score_train