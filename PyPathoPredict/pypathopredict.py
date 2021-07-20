import pandas as pd
import numpy as np
from sklearn.svm import SVC
from os import listdir, path
import pickle

with open("final_model.pkl","r") as f:
    model = pickle.load(f)


def pathopredict(dfIncomplete):

    dfPre1 = pd.DataFrame({"Prediction": model.predict(dfIncomplete[["iso_point","pdel"]])})
    dfPre2 = pd.DataFrame(model.predict_proba(dfIncomplete[["iso_point","pdel"]]),
                       index=pd.RangeIndex(start=0, stop=len(dfPre1)),
                       columns=['Confidence Negative', 'Confidence Positive'])
    dfPre3 = pd.DataFrame({"Index": dfIncomplete["index_name"]})

    dfPre12 = dfPre1.join(dfPre2)

    return dfPre3.join(dfPre12)


if __name__ == '__main__':
    
    dataframe_dir = {} 
    
    #getting input csvs
    for file in listdir("Input"):
        dataframe_dir[file] = pd.read_csv(file)
    
    #outputing csvs
    for key, value in dataframe_dir.items:
        #running the function
        output = pathopredict(value)
        
        output_filename = key.split(".")[0] + "_output.csv"
        
        output.to_csv(path.join("Output", output_filename))