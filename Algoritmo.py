import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input



np.random.seed(42)

n = 1000

data = {
"acetone": np.random.normal(1.2,0.4,n),
"ethanol": np.random.normal(0.8,0.3,n),
"ammonia": np.random.normal(2.5,0.9,n),
"isoprene": np.random.normal(0.5,0.2,n),
"hydrogen_sulfide": np.random.normal(0.3,0.15,n),
"methane": np.random.normal(1.0,0.5,n),
"temperature": np.random.normal(34,0.8,n),
"humidity": np.random.normal(85,5,n)
}

df = pd.DataFrame(data)

score = (
0.4*df["ammonia"] +
0.2*df["acetone"] +
0.2*df["hydrogen_sulfide"] +
0.1*df["methane"]
)

df["cirrhosis"] = (score > score.mean()).astype(int)



X = df.drop("cirrhosis",axis=1)
y = df["cirrhosis"]



scaler = StandardScaler()
X = scaler.fit_transform(X)



X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42
)



model = Sequential()

model.add(Input(shape=(X_train.shape[1],)))

model.add(Dense(16,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))

model.add(Dense(1,activation='sigmoid'))

model.compile(
optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy']
)

history = model.fit(
X_train,
y_train,
epochs=50,
batch_size=16,
validation_data=(X_test,y_test)
)



feature_model = Model(
inputs=model.inputs,
outputs=model.layers[-2].output
)

X_train_feat = feature_model.predict(X_train)
X_test_feat = feature_model.predict(X_test)

-

svm = SVC(kernel='rbf',probability=True)

svm.fit(X_train_feat,y_train)

y_pred = svm.predict(X_test_feat)



cm = confusion_matrix(y_test,y_pred)

plt.figure()

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Real")

plt.colorbar()

plt.show()



y_prob = svm.predict_proba(X_test_feat)[:,1]

fpr,tpr,_ = roc_curve(y_test,y_prob)

roc_auc = auc(fpr,tpr)

plt.figure()

plt.plot(fpr,tpr,label="AUC="+str(round(roc_auc,3)))

plt.legend()

plt.title("ROC Curve")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.show()



plt.figure()

plt.plot(history.history['loss'],label="Training")

plt.plot(history.history['val_loss'],label="Validation")

plt.legend()

plt.title("Backpropagation Training")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.show()



pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_train_feat)

svm_vis = SVC(kernel='rbf')

svm_vis.fit(X_pca,y_train)



x_min,x_max = X_pca[:,0].min()-1, X_pca[:,0].max()+1
y_min,y_max = X_pca[:,1].min()-1, X_pca[:,1].max()+1

xx,yy = np.meshgrid(
np.linspace(x_min,x_max,300),
np.linspace(y_min,y_max,300)
)

Z = svm_vis.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()

plt.contourf(xx,yy,Z,alpha=0.3)

plt.scatter(
X_pca[:,0],
X_pca[:,1],
c=y_train
)

plt.title("SVM Kernel Decision Boundary")

plt.xlabel("PCA Component 1")

plt.ylabel("PCA Component 2")

plt.show()