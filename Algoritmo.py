import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from sklearn.decomposition import PCA

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

np.random.seed(42)

plt.rcParams["figure.figsize"] = (8, 6)

sns.set_theme(style="whitegrid")


# ============================================================
# 2. CARGAR DATASET
# ============================================================

# Si voc_dataset_1000.csv está en la misma carpeta que Algoritmo.py
df = pd.read_csv("voc_dataset_1000.csv")

print("\n==========================================")
print("DATASET")
print("==========================================")

print("Filas:", df.shape[0])
print("Columnas:", df.shape[1])

print("\nVariables:")
print(df.columns.tolist())


# ============================================================
# 3. VARIABLES COV
# ============================================================

voc_columns = [
    "acetone_ppm",
    "ethanol_ppm",
    "ammonia_ppm",
    "isoprene_ppm",
    "hydrogen_sulfide_ppm",
    "methane_ppm"
]


# Variables clínicas/adicionales
additional_columns = []

if "temperature" in df.columns:
    additional_columns.append("temperature")

if "humidity" in df.columns:
    additional_columns.append("humidity")


feature_columns = voc_columns + additional_columns


# ============================================================
# 4. COMPROBAR VARIABLES
# ============================================================

missing_columns = [
    col for col in feature_columns + ["cirrhosis"]
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Faltan las siguientes columnas en el dataset: "
        f"{missing_columns}"
    )


# ============================================================
# 5. ELIMINAR VALORES FALTANTES
# ============================================================

df = df[
    feature_columns + ["cirrhosis"]
].dropna().copy()


# ============================================================
# 6. CONVERTIR VARIABLE OBJETIVO
# ============================================================

df["cirrhosis"] = pd.to_numeric(
    df["cirrhosis"]
).astype(int)


# ============================================================
# 7. DISTRIBUCIÓN DE CLASES
# ============================================================

print("\n==========================================")
print("DISTRIBUCIÓN DE CLASES")
print("==========================================")

print(
    df["cirrhosis"].value_counts()
)


# ============================================================
# 8. GRÁFICA 1
# DISTRIBUCIÓN DE COVs
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10)
)

axes = axes.flatten()


for idx, voc in enumerate(voc_columns):

    sns.violinplot(
        data=df,
        x="cirrhosis",
        y=voc,
        ax=axes[idx],
        inner="quartile"
    )

    nombre = (
        voc
        .replace("_ppm", "")
        .replace("_", " ")
        .title()
    )

    axes[idx].set_title(
        nombre,
        fontsize=12,
        fontweight="bold"
    )

    axes[idx].set_xlabel(
        "Estado de cirrosis"
    )

    axes[idx].set_ylabel(
        "Concentración (ppm)"
    )

    axes[idx].set_xticks([0, 1])

    axes[idx].set_xticklabels(
        [
            "Sin cirrosis",
            "Con cirrosis"
        ]
    )


plt.suptitle(
    "Distribución de biomarcadores COV según el estado de cirrosis",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "01_distribucion_cov.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# 9. GRÁFICA 2
# RELACIONES ENTRE COVs
# ============================================================

pairs = [
    ("acetone_ppm", "ammonia_ppm"),
    ("acetone_ppm", "methane_ppm"),
    ("ammonia_ppm", "methane_ppm")
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)


for idx, (voc1, voc2) in enumerate(pairs):

    for clase in [0, 1]:

        subset = df[
            df["cirrhosis"] == clase
        ]

        if clase == 0:

            color = "#2ecc71"
            label = "Sin cirrosis"

        else:

            color = "#e74c3c"
            label = "Con cirrosis"

        axes[idx].scatter(
            subset[voc1],
            subset[voc2],
            alpha=0.6,
            s=35,
            color=color,
            label=label,
            edgecolors="black",
            linewidth=0.4
        )


    nombre1 = (
        voc1
        .replace("_ppm", "")
        .replace("_", " ")
        .title()
    )

    nombre2 = (
        voc2
        .replace("_ppm", "")
        .replace("_", " ")
        .title()
    )


    axes[idx].set_xlabel(
        nombre1 + " (ppm)",
        fontweight="bold"
    )

    axes[idx].set_ylabel(
        nombre2 + " (ppm)",
        fontweight="bold"
    )

    axes[idx].set_title(
        f"{nombre1} vs. {nombre2}",
        fontweight="bold"
    )

    axes[idx].legend()


plt.suptitle(
    "Relaciones entre los principales biomarcadores COV",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "02_relaciones_cov.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# 10. VARIABLES DE ENTRADA
# ============================================================

X = df[
    feature_columns
]

y = df[
    "cirrhosis"
]


# ============================================================
# 11. DIVISIÓN 80/20 ESTRATIFICADA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n==========================================")
print("DIVISIÓN DEL DATASET")
print("==========================================")

print(
    "Entrenamiento:",
    X_train.shape
)

print(
    "Prueba:",
    X_test.shape
)


# ============================================================
# 12. ESTANDARIZACIÓN
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# ============================================================
# 13. RED NEURONAL ARTIFICIAL
# ============================================================

model = Sequential()

model.add(
    Input(
        shape=(X_train.shape[1],)
    )
)

model.add(
    Dense(
        16,
        activation="relu"
    )
)

model.add(
    Dense(
        12,
        activation="relu"
    )
)

model.add(
    Dense(
        8,
        activation="relu"
    )
)

# Capa de características
model.add(
    Dense(
        4,
        activation="relu"
    )
)

# Salida
model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 14. ENTRENAMIENTO
# ============================================================

print("\n==========================================")
print("ENTRENAMIENTO ANN")
print("==========================================")


history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(
        X_test,
        y_test
    ),
    verbose=1
)


# ============================================================
# 15. EXTRACCIÓN DE CARACTERÍSTICAS
# ============================================================

feature_model = Model(
    inputs=model.inputs,
    outputs=model.layers[-2].output
)


X_train_feat = feature_model.predict(
    X_train,
    verbose=0
)

X_test_feat = feature_model.predict(
    X_test,
    verbose=0
)


print("\n==========================================")
print("CARACTERÍSTICAS APRENDIDAS")
print("==========================================")

print(
    "Entrenamiento:",
    X_train_feat.shape
)

print(
    "Prueba:",
    X_test_feat.shape
)


# ============================================================
# 16. SVM-RBF
# ============================================================

print("\n==========================================")
print("ENTRENAMIENTO SVM-RBF")
print("==========================================")


svm = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)


svm.fit(
    X_train_feat,
    y_train
)


# ============================================================
# 17. PREDICCIÓN
# ============================================================

y_pred = svm.predict(
    X_test_feat
)

y_prob = svm.predict_proba(
    X_test_feat
)[:, 1]


# ============================================================
# 18. MÉTRICAS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)


print("\n==========================================")
print("RESULTADOS ANN-SVM-RBF")
print("==========================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-Score  : {f1:.4f}"
)

print(
    f"AUC       : {roc_auc:.4f}"
)


# ============================================================
# 19. MATRIZ DE CONFUSIÓN
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


TN, FP, FN, TP = cm.ravel()


print("\n==========================================")
print("MATRIZ DE CONFUSIÓN")
print("==========================================")

print(
    "Verdaderos negativos:",
    TN
)

print(
    "Falsos positivos:",
    FP
)

print(
    "Falsos negativos:",
    FN
)

print(
    "Verdaderos positivos:",
    TP
)


# ============================================================
# 20. GRÁFICA MATRIZ DE CONFUSIÓN
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    ax=ax,
    annot_kws={
        "size": 18,
        "weight": "bold"
    }
)


ax.set_xlabel(
    "Predicción",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "Valor real",
    fontsize=12,
    fontweight="bold"
)

ax.set_title(
    "Matriz de confusión del modelo ANN-SVM-RBF",
    fontsize=14,
    fontweight="bold"
)

ax.set_xticklabels(
    [
        "Sin cirrosis",
        "Con cirrosis"
    ]
)

ax.set_yticklabels(
    [
        "Sin cirrosis",
        "Con cirrosis"
    ],
    rotation=90
)


plt.tight_layout()

plt.savefig(
    "03_matriz_confusion.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# 21. CURVA ROC
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "Tasa de falsos positivos"
)

plt.ylabel(
    "Tasa de verdaderos positivos"
)

plt.title(
    "Curva ROC del modelo ANN-SVM-RBF",
    fontweight="bold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "04_curva_ROC.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# 22. CURVA DE ENTRENAMIENTO
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.plot(
    history.history["loss"],
    label="Entrenamiento"
)

plt.plot(
    history.history["val_loss"],
    label="Validación"
)

plt.xlabel(
    "Épocas"
)

plt.ylabel(
    "Pérdida"
)

plt.title(
    "Entrenamiento de la Red Neuronal mediante Backpropagation",
    fontweight="bold"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "05_entrenamiento_ANN.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# 23. PCA PARA VISUALIZACIÓN 3D
# ============================================================

X_all_feat = np.vstack(
    [
        X_train_feat,
        X_test_feat
    ]
)

y_all = np.concatenate(
    [
        y_train.to_numpy(),
        y_test.to_numpy()
    ]
)


pca = PCA(
    n_components=3
)


X_3D = pca.fit_transform(
    X_all_feat
)


# ============================================================
# 24. VARIANZA PCA
# ============================================================

variance = (
    pca.explained_variance_ratio_
)


print("\n==========================================")
print("PCA")
print("==========================================")

print(
    f"PC1: {variance[0] * 100:.2f}%"
)

print(
    f"PC2: {variance[1] * 100:.2f}%"
)

print(
    f"PC3: {variance[2] * 100:.2f}%"
)

print(
    f"Total: {variance.sum() * 100:.2f}%"
)


# ============================================================
# 25. GRÁFICA 3D
# ============================================================

fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


clase_0 = (
    y_all == 0
)

clase_1 = (
    y_all == 1
)


ax.scatter(
    X_3D[clase_0, 0],
    X_3D[clase_0, 1],
    X_3D[clase_0, 2],
    label="Sin cirrosis",
    alpha=0.7,
    s=35
)


ax.scatter(
    X_3D[clase_1, 0],
    X_3D[clase_1, 1],
    X_3D[clase_1, 2],
    label="Con cirrosis",
    alpha=0.7,
    s=35
)


ax.set_xlabel(
    "Componente principal 1 (PC1)"
)

ax.set_ylabel(
    "Componente principal 2 (PC2)"
)

ax.set_zlabel(
    "Componente principal 3 (PC3)"
)


ax.set_title(
    "Separación tridimensional de las clases en el espacio aprendido por la ANN",
    fontweight="bold"
)


ax.legend()


plt.tight_layout()

plt.savefig(
    "06_separacion_3D_ANN.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# 26. REPORTE FINAL
# ============================================================

print("\n==========================================")
print("PROCESO FINALIZADO")
print("==========================================")

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall:",
    round(recall * 100, 2),
    "%"
)

print(
    "F1-Score:",
    round(f1 * 100, 2),
    "%"
)

print(
    "AUC:",
    round(roc_auc, 4)
)

print("\nGráficas generadas:")
print("01_distribucion_cov.png")
print("02_relaciones_cov.png")
print("03_matriz_confusion.png")
print("04_curva_ROC.png")
print("05_entrenamiento_ANN.png")
print("06_separacion_3D_ANN.png")