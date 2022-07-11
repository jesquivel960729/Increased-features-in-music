import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.cluster import KMeans
from keras.datasets import cifar10
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV


MUSIC_PATH = "./data_new(1).csv"

def load_music_data(music_path = MUSIC_PATH):
    csv_path = os.path.join(music_path)
    return pd.read_csv(csv_path)

artistas = load_music_data()
dataframe = pd.DataFrame(artistas)


################################FACTORIZAR GENEROS###############
df = artistas[["Name","Genres"]]
tifidf = TfidfVectorizer(stop_words="english")
tifidf_matrix = tifidf.fit_transform(df["Genres"])
df_dtm = pd.DataFrame(tifidf_matrix.toarray(),index=df["Name"].values,columns=tifidf.get_feature_names())

valor = []
#print(df_dtm)
M = tifidf_matrix.toarray()
for i in range(len(M)):
    result = 0
    for j in range(len(M[i])):
        result = result + float(M[i][j])
    
    valor.append(result)
#################################################################
dfw = artistas[["Name","Writers"]]
tifidfw = TfidfVectorizer(stop_words="english")
tifidfw_matrix = tifidfw.fit_transform(dfw["Writers"])
df_dtmw = pd.DataFrame(tifidfw_matrix.toarray(),index=df["Name"].values,columns=tifidfw.get_feature_names())

valorw = []
print(df_dtmw)
Mw = tifidfw_matrix.toarray()
#for i in range(len(Mw)):
#    result = 0
#    for j in range(len(Mw[i])):
#        result = result + float(Mw[i][j])    
#    valorw.append(result)
#
#NUEVO = pd.DataFrame({'Name': artistas.Name,'Features': artistas.Features ,'Genres': valor,'Writers': valorw, 'AccRank': dataframe['AccRank']})

def format_text(txt):
    x = txt.replace("['", "")
    x = x.replace("']", "")
    return x.split("', '")
#NUEVO.to_csv("./resumen.csv")

#def build_data_set(data_authors, author):
#    author = "ed sheeran"
#    artist = artistas.loc[artistas["Name"]==author]
#    index = artist.index.to_numpy()[0]
#    dataset = []
#    for i in range(len(NUEVO)):
#        if NUEVO.Name[i] != author:
#            name_author = NUEVO.Name[i]
#            feat_author = 1 if name_author in format_text(artistas.Features[index]) else 0
#            dataset.append([name_author,feat_author,NUEVO.Genres[i],NUEVO.Writers[i],NUEVO.AccRank[i]])
#
#    resultadofinal = pd.DataFrame(dataset, columns = ["Name", "Features", "Genres", "Writers","AccRank"])
#    #resultadofinal.to_csv("./resumen2.csv")
#    return resultadofinal
#
#data_set = build_data_set(NUEVO,"ed sheeran")
#
## Creating the test set and train set
#train_set, test_set = train_test_split(data_set, test_size=0.2, random_state=7)
## Revert to a clean training set and separate predictors and labels
#authors_labels = train_set["Features"].copy()
#authors_set = train_set.drop("Features", axis=1)
#print(authors_set.keys())
## Also need to drop "release_date", "id", "name" and "artists"
#authors_set = authors_set.drop("Name", axis=1)
## Transformation Pipelines and Feature Scaling
#num_pipeline = Pipeline([
#    ('imputer', SimpleImputer(strategy="median")),
#    ('std_scaler', StandardScaler())
#])
#authors_prepared = num_pipeline.fit_transform(authors_set)
## Building a Random Forest Regressor Model
#print("Training a Random Forest Regressor Model...")
#
#forest_reg = RandomForestRegressor(n_estimators=100, random_state=0)
#forest_reg.fit(authors_prepared, authors_labels)
#
## Preparing the test set so we can check the model against it.
#test_set = test_set.drop("Name", axis=1)
#y_test = test_set["Features"].copy()
#x_test = test_set.drop("Features", axis=1)
## Evaluate the model on the test set
#final_model = forest_reg
#x_test_prepared = num_pipeline.transform(x_test)
#final_predictions = final_model.predict(x_test_prepared)
#final_mse = mean_squared_error(y_test, final_predictions)
#final_r2score = r2_score(y_test, final_predictions)
#print(final_r2score)
#print(final_mse)
## Can print out the first 100 entries to check the performance of the model 
#for index in range(len(final_predictions)): 
#    pred = final_predictions[index]
#    actual = y_test.iloc[index]
#    #artist = x_test.iloc[index].to_numpy()
#    #print(artist)
#    #print("Actual / Predicted: {:.4f} / {:.4f}".format(actual, pred))
#
#def rfr_model(X, y):
#    # Perform Grid-Search
#    gsc = GridSearchCV(
#        estimator=RandomForestRegressor(),
#        param_grid={
#            'max_depth': range(3,7),
#            'n_estimators': (10, 50, 100, 1000),
#        },
#        cv=5, scoring='neg_mean_squared_error', verbose=0,                         
#        n_jobs=-1)
#    
#    grid_result = gsc.fit(X, y)
#    best_params = grid_result.best_params_
#    
#    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],                               random_state=False, verbose=False)
#    # Perform K-Fold CV
#    print(rfr.score())
#    scores = cross_val_score(rfr, X, y, cv=10)
#
#    predictions = cross_val_predict(rfr, X, y, cv=10)
#
#    #print(scores)
#    return scores
#
#data_set = build_data_set(NUEVO, "ed sheeran")
#authors_name = data_set["Name"].copy()
#data = data_set.drop("Name", axis=1)
#variable_dependence = data["Features"].copy()
#features = data.drop("Features", axis=1)
# #print(variable_dependence)
# #print(features)
#rfr_model(features, variable_dependence)
#
#