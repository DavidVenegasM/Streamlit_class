import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Utils:

    # Carga la información del path deseado:
    def load_from_csv(self,path):
        return pd.read_csv(path)

    # Realiza los cambios a las variables que deseamos modificar:
    def features_target(self, dataset, drop_col, cols_wanted, target):
        w = dataset.drop(drop_col, axis = 1)
        X = pd.get_dummies(w,cols_wanted)
        y = dataset[target]
        return X, y

    # Imprime la gráfica de la columna deseada
    def grafica_barras(self, dataset, columna):
        sns_plot = sns.distplot(dataset[columna])
        plt.title(f'{columna}')
        fig = sns_plot.get_figure()
        fig.savefig('C:/Users/David/Downloads/Streamlit_Clase/Streamlit_Clase/out/grafica_1.png')
   
