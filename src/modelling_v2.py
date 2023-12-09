import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import  confusion_matrix, auc, roc_curve, roc_auc_score, precision_score, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# V2
# ----------------------------------------------------------------
# Predictive Modelling
# ----------------------------------------------------------------

# ROC Curve: Area Under the Curve for evaluation
def auc_roc_plot(y_test, y_preds):
    fpr, tpr, thresholds = roc_curve(y_test,y_preds)
    roc_auc = auc(fpr, tpr)
    print("Predicted accuracy: " + str(roc_auc))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

seed = 42

def logistic_regression(df):
    train_set_lr, test_set_lr = train_test_split(df, test_size = 0.2, random_state = seed)
    # train data
    X_train_lr = train_set_lr.drop(['status'], axis = 1)
    y_train_lr = train_set_lr['status']
    # test data
    X_test_lr = test_set_lr.drop(['status'], axis = 1)
    y_test_lr = test_set_lr['status']

    # Normalizing the train and test data
    scaler_lr = MinMaxScaler()
    features_names = X_train_lr.columns
    X_train_lr = scaler_lr.fit_transform(X_train_lr)
    X_train_lr = pd.DataFrame(X_train_lr, columns = features_names)
    X_test_lr = scaler_lr.transform(X_test_lr)
    X_test_lr = pd.DataFrame(X_test_lr, columns = features_names)

    lr = LogisticRegression(max_iter = 1000, solver = 'lbfgs', random_state = seed, class_weight = 'balanced' )
    parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}
    clf_lr = GridSearchCV(lr, parameters, cv = 5).fit(X_train_lr, y_train_lr)

    y_preds_lr = clf_lr.predict_proba(X_test_lr)[:,1]

    auc_roc_plot(y_test_lr, y_preds_lr)

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test_lr == 1, y_preds_lr > 0.5).ravel()
    print("Confusion Matrix:")
    print(tn, fp, fn, tp)
    return clf_lr, scaler_lr

def predict_status_lr(logistic_regression_model, df):
    # Use the trained model to predict probabilities
    predicted_probs = logistic_regression_model.predict_proba(df)[:, 1]

    return predicted_probs

def knn(df):
    # dividing the dataset in train (80%) and test (20%)
    train_set_knn, test_set_knn = train_test_split(df, test_size = 0.2, random_state = seed)
    X_train_knn = train_set_knn.drop(['status'], axis = 1)
    y_train_knn = train_set_knn['status']
    X_test_knn = test_set_knn.drop(['status'], axis = 1)
    y_test_knn = test_set_knn['status']

    # normalizing train and test data
    scaler_knn = MinMaxScaler()
    features_names = X_train_knn.columns
    X_train_knn = scaler_knn.fit_transform(X_train_knn)
    X_train_knn = pd.DataFrame(X_train_knn, columns = features_names)
    X_test_knn = scaler_knn.transform(X_test_knn)
    X_test_knn = pd.DataFrame(X_test_knn, columns = features_names)

    # time
    best_k = 0
    max_acc = 0
    for k in range(1, 200, 5):
        k = k + 1
        knn = KNeighborsClassifier(n_neighbors = k).fit(X_train_knn, y_train_knn)
        acc = knn.score(X_test_knn, y_test_knn)
        # get the maximum accuracy K
        if acc > max_acc:
            max_acc = acc
            best_k = k
        # print('Accuracy for k =', k, ' is:', acc)
    print('Accuracy for best k =', best_k, ' is:', max_acc)

    knn = KNeighborsClassifier(n_neighbors = best_k, weights='uniform').fit(X_train_knn, y_train_knn)
    y_preds_knn = knn.predict(X_test_knn)
    auc_roc_plot(y_test_knn, y_preds_knn)

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test_knn == 1, y_preds_knn > 0.5).ravel()
    print("Confusion Matrix:")
    print(tn, fp, fn, tp)

def decision_trees(df):
    # split into features and target variable
    X = df.drop('status', axis=1)
    y = df['status']
    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # train decision trees model
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    # make the predictions
    y_pred = dt_model.predict(X_test)
    y_proba = dt_model.predict_proba(X_test)[:, 1]
    # evaluate the model
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    auc_roc_plot(y_test, y_pred)

    print(f'AUC: {auc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Accuracy: {accuracy:.4f}')

def random_forest(df):
    # split into features and target variable
    X = df.drop('status', axis=1)
    y = df['status']
    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # train the Random Forest model
    # !!! n_estimators is the number of trees in the forest
    # experimentally deduced that 500 is the best value, if we increase over 1000 we will have a decrease in precision and accuracy
    rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
    rf_model.fit(X_train, y_train)
    # make predictions on the test set
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    # evaluate the performance
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    auc_roc_plot(y_test, y_pred)

    print(f'AUC: {auc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Accuracy: {accuracy:.4f}')

def NN(df):
    X = df.drop('status', axis=1)
    y = df['status']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # standardize the input features (optional but often recommended for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # build a simple neural network model
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.25)

    # make predictions on the test set
    y_proba = model.predict(X_test_scaled)
    y_pred = (y_proba > 0.5).astype(int)

    # evaluate the performance
    auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    auc_roc_plot(y_test, y_pred)

    print(f'AUC: {auc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
