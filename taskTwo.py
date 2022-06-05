import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM
import sys

# Data Path
DATA_PATH = sys.argv[1]+"/data"

FUNCTIONAL_PATH = DATA_PATH + '/js/functionalJS/'
TRACKING_PATH = DATA_PATH + '/js/trackingJS/'
FUNCTIONAL_FILES = os.listdir(FUNCTIONAL_PATH)
TRACKING_FILES = os.listdir(TRACKING_PATH)
js_df = pd.DataFrame(
    data={
        'Type': ['functional' for i in FUNCTIONAL_FILES] + ['tracking' for i in TRACKING_FILES],
        "File": FUNCTIONAL_FILES + TRACKING_FILES})

# Task 2
# Pre-processing
print("Preprocessing Data...")

# Remove any files that are not utf-8 encoded
for func_file in FUNCTIONAL_FILES:
    try:
        f = open(FUNCTIONAL_PATH + func_file, "rb")
        f.read().decode('utf-8')
    except UnicodeDecodeError:
        FUNCTIONAL_FILES.remove(func_file)

for track_file in TRACKING_FILES:
    try:
        f = open(TRACKING_PATH + track_file, "rb")
        f.read().decode("utf-8")
    except UnicodeDecodeError:
        TRACKING_FILES.remove(track_file)

# Task 2.1
print("Fitting TF-IDF models")
# train the TD-IDF model for tracking js files
tracking_files = [TRACKING_PATH + x for x in TRACKING_FILES]
tracking_vectorizer = TfidfVectorizer(input="filename")
X_tracking = tracking_vectorizer.fit_transform(tracking_files)



# Task 2.2
# Assemble data frame
# Create dataframes with label, file name and feature
# *** need to do seperately due to TF-IDF training seperately
# concat both dataframes
# Shuffle to reduce bias when splitting
js_df_func = pd.DataFrame(
    data={
        'Label': ['functional' for i in FUNCTIONAL_FILES],
        "File": FUNCTIONAL_FILES})
js_df_track = pd.DataFrame(
    data={
        'Label': ['tracking' for i in TRACKING_FILES],
        "File": TRACKING_FILES})
JS_DF = pd.concat([js_df_func, js_df_track]).sample(frac=1)

# Train the One Class SVM and fit the tracking TF-IDF features
print("Training One Class SVM...")
clf_ocsvm = OneClassSVM(gamma='scale', nu=0.2).fit(X_tracking)


# Task 2.3

# calculate the accuracy of the model across functional and tracking scripts
def scoreOCSVM(preds, labels):
    count = 0
    for i in range(len(preds)):
        if preds[i] == labels[i]:
            count += 1
    return count / len(preds)


# Train and fit OCSVM
def OCSVM(gamma, nu):
    # Train the OCSVM model
    clf_ocsvm = OneClassSVM(gamma=gamma, nu=nu).fit(X_tracking)
    # Vectorise the tracking files
    trackingjs_functifidfs = tracking_vectorizer.transform([TRACKING_PATH + file for file in js_df_track["File"]])
    preds_tracking_functsidf = clf_ocsvm.predict(trackingjs_functifidfs)
    # vectorise the functional files
    trackingjs_functifidfs = tracking_vectorizer.transform([FUNCTIONAL_PATH + file for file in js_df_func["File"]])
    preds_functional_functsidf = clf_ocsvm.predict(trackingjs_functifidfs)
    # Generate labels and predictions
    labels = [1 for i in preds_tracking_functsidf] + [-1 for i in preds_functional_functsidf]
    preds = [i for i in preds_tracking_functsidf] + [i for i in preds_functional_functsidf]
    return scoreOCSVM(preds, labels)


# Execute the OCSVM
print("Executing Base OCSVM: " + str(OCSVM("scale", 0.2)))

# Tuning the model
print("\nTuning the SVM Model...")
nus = np.arange(0.2, 0.8, 0.2)
gammas = np.arange(0.2, 0.8, 0.2)
scores = []
for nu in nus:
    for gamma in gammas:
        score = [OCSVM(gamma, nu), nu, gamma]
        scores.append(score)
    print(str(nu * 100) + "%")
print("Done tuning")

best = 0
best_score = []
for score in scores:
    if score[0] > best:
        best = score[0]
        best_score = score
print("Best Score: " + str(best_score[0]) + ", Nu: " + str(best_score[1]) + ", Gamma: " + str(best_score[2]))
