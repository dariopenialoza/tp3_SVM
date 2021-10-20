import matplotlib as plt
import numpy as np
import pandas as pd
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

cielo = Image.open('cielo.jpg')
cieloA = np.array(cielo)
cieloM = np.reshape(cieloA,(-1,3))

pasto = Image.open('pasto.jpg')
pastoA = np.array(pasto)
pastoM = np.reshape(pastoA,(-1,3))

vaca = Image.open('vaca.jpg')
vacaA = np.array(vaca)
vacaM = np.reshape(vacaA,(-1,3))

cieloDF = pd.DataFrame(cieloM,columns=list('rgb'))
cieloDF = cieloDF.assign(Tipo=1)
pastoDF = pd.DataFrame(pastoM,columns=list('rgb'))
pastoDF = pastoDF.assign(Tipo=2)
vacaDF = pd.DataFrame(vacaM,columns=list('rgb'))
vacaDF = vacaDF.assign(Tipo=2)

dataset = cieloDF.append(pastoDF,ignore_index=True)
dataset = dataset.append(vacaDF,ignore_index=True)

# Preprocesado y modelado
# ==============================================================================
color = dataset.drop(columns = 'Tipo')
tipo = dataset['Tipo']

color_train, color_test, tipo_train, tipo_test = train_test_split(
                                        color,
                                        tipo,
                                        train_size   = 0.7,
                                        random_state = 1234,
                                        shuffle      = True
                                    )
# Creación del modelo SVM lineal
# ==============================================================================
model = SVC(C = 100, kernel = 'linear', random_state=123)
modelo = model.fit(color_train, tipo_train)

#Prediciendo con el testing set
#grid = np.vstack([color_test.r.ravel(),color_test.g.ravel(),color_test.b.ravel()]).T

# Predicción valores grid
predicciones = modelo.predict(color_test)

# Creamos la figura
fig2 = plt.figure()
ax2 = fig2.add_subplot( 111, projection='3d')

x = color_test.r
y = color_test.g
z = color_test.b
ax2.scatter3D(x, y, z,
              c=predicciones, marker='o',
              depthshade=True, s=4,
              cmap=plt.cm.tab20,
              edgecolors='gray',
              linewidth=0.1,)

ax2.set_title("Es cielo o no, C=100, Kernel='Linear'")
ax2.set_xlabel('R')
ax2.set_ylabel('G')
ax2.set_zlabel('B')

# Vectores soporte
ax2.scatter3D(
    modelo.support_vectors_[:, 0],
    modelo.support_vectors_[:, 1],
    modelo.support_vectors_[:, 2],
    s=40, linewidth=1,
    facecolors='none', edgecolors='red'
)

#Hiperplano
z = lambda x,y: (-modelo.intercept_[0]-modelo.coef_[0][0]*x -modelo.coef_[0][1]*y) / modelo.coef_[0][2]

tmp = np.linspace(0,255,25)
x,y = np.meshgrid(tmp,tmp)

ax2.plot_surface(x, y, z(x,y),alpha=0.5)

plt.show()

# Accuracy de test del modelo # ==============================================================================
accuracy = accuracy_score(
            y_true    = tipo_test,
            y_pred    = predicciones,
            normalize = True
           )
print("")
print(f"El accuracy de test es: {100*accuracy} %")

# Matriz de confusión de las predicciones de test# ==============================================================================
confusion_matrix = pd.crosstab(
    tipo_test.ravel(),
    predicciones,
    rownames=['Real'],
    colnames=['Predicción']
)
print(confusion_matrix)
