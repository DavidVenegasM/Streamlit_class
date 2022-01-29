# Cargamos las clases de los módulos que creamos:

from utils import Utils
from models import Models

# Es el Script PRINCIPAL
if __name__ == "__main__":
    
    # Inicializamos la clase

    utils = Utils()
    model = Models()
 
    # Cargamos la info
    data = utils.load_from_csv('C:/Users/David/Downloads/Streamlit_Clase/Streamlit_Clase/in/income.csv')

    # Mostramos un pedacito de nuestro dataset:
    print('Muestra del dataset')
    print(data.head(5))

    ## Creación de las graficas:
    utils.grafica_barras(data, 'age')


    ## Creación del modelo y evaluación
    dropear = ['fnlwgt','capital-gain','capital-loss','income','native-country','income_bi']
    dummies = ['workclass','education','marital-status','occupation','relationship','race','sex']
    target = 'income_bi'

    X , y = utils.features_target(data,dropear,dummies,target)

    resultados = model.tree_training(X,y)
    print(f'Score train: {resultados[1]}')
    print(f'Score test: {resultados[0]}')