import streamlit as st
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.model_selection import train_test_split

#Carga de info

df = pd.read_csv('./in/income.csv')
st.image('pig.jpg')

siteHeader = st.container()
with siteHeader:
    st.title('Modelo de evaluación de ingresos')
    st.markdown('el objetivo de este proyecto es proveer una herramienta que nos permita predericr sí una persona ganará más op menos de $50k anuales')


daviz = st.container()
with daviz:
    st.subheader('Explroación de la data:')
    st.caption('¿Habrá diferencia entre hombres y mujeres?')

    filtro = st.radio('Elige la opción a graficar',['Ambos', 'Hombres', 'Mujeres'])

    data = df.copy()
    if filtro == 'Mujeres':
        data = df[df.sex == ' Female']
    elif filtro == 'Hombres':
        data = df[df.sex == ' Male']
    
    st.text('Distribución por ocupación')
    st.bar_chart(data.occupation.value_counts())

    st.text('Distribución por edad')
    st.bar_chart(data.age.value_counts())

    col1, col2 = st.columns(2)    
    col1.text('Distribución por ocupación')
    col1.bar_chart(data.occupation.value_counts())
    
    col2.text('Distribución por edad')
    col2.bar_chart(data.age.value_counts())

newFeatures = st.container()
with newFeatures:
    st.subheader('Selección de variables:')
    st.markdown('De manera inicial, el modelo trabaja con la edad, el promedio de horas laborales por semana y las variables: **race, sex, workclass y education.**')
    st.text('Quieres considerar alguna otra varialbe? Selecciona la que quieras:')

    optional_cols= ['education-num','marital-status','occupation','relationship']

    options = st.multiselect('Variables que se añadiran al modelo: ', optional_cols)

principal_columns = ['race','sex','workclass','education']
drop_columns = ['income','fnlwgt','capital-gain','capital-loss','native-country','income_bi']

if len(options) != 0:
    principal_columns = principal_columns + options
    drop_columns = drop_columns + [i for i in optional_cols if i not in options]
else:
    drop_columns = drop_columns + optional_cols


modelTrain = st.container()
with modelTrain:
    st.subheader('Entrenamiento del modelo')
    st.text('En esta sección puedes cambiar el hiperparámetro del modelo')

    max_depth = st.slider('Cual es el valor de profundidad para el modelo?', min_value = 2, max_value = 10, value = 7, step = 1)

    # Definimos nuestras variables:   
    Y = df['income_bi']
    df = df.drop(drop_columns, axis=1)
    
    X = pd.get_dummies(df, columns = principal_columns)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state= 15)

    # Modelo
    t = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = max_depth)
    model = t.fit(x_train, y_train)

    # Performance del modelo
    score_train = model.score(x_train, y_train)
    score_test = model.score(x_test, y_test)

    col1, col2 = st.columns(2)
    with col1:
        st.text(f'Train Score:{round(score_train *100,2)}%')
    with col2:
        st.text(f'Test Score:{round(score_test *100,2)}%')

userTesteo = st.container()
with userTesteo:
    st.subheader('Prueba el modelo')
    st.markdown('Mezcla las opciones para descubir si ganarías ás o menos de $50K')

    dict_test = {}
    for col in ['age','hours-per-week'] + principal_columns:
        dict_test[col] = st.selectbox(f'{col}:',
        list(df[col].unique()))
    
    # Formato del input
    cols_x = list(X.columns)
    df_form = pd.DataFrame(columns = cols_x)

    # Opciones elegidas
    x_pred_list = [dict_test[col] for col in dict_test.keys()]
    index = list(dict_test.keys())
    df_pred = pd.DataFrame(x_pred_list, index = index).T
    x_pred = pd.get_dummies(df_pred, columns = principal_columns) 
    st.caption('Muestra:')
    st.dataframe(x_pred)
    
    # Cambiamos la forma
    for col in x_pred.columns:
          df_form[col] = x_pred[col]

    df_form.fillna(0, inplace = True)

    st.caption('Selecciona esta casilla par calcular el resultado:')
    calcular = st.checkbox('Calcular')

    if calcular:
        proba = model.predict_proba(df_form)[0]
        resultado = model.predict(df_form)
        if resultado == 0:
            st.markdown(f'la persona no ganara mas de 50k con probabilidad de {round(proba[0]*100,2)}%')
        else:
            st.markdown(f'la persona ganara mas de 50k con probabilidad de {round(proba[1]*100,2)}%')

    
    
