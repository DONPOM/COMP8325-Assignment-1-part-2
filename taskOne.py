import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
import sys

# Data Path
DATA_PATH = sys.argv[1]+"/data"

print("\nTask 1")

# Task 1 ==========================================================================================================================================
data = pd.read_csv(DATA_PATH+'/server-log.txt', sep=" ")
data.rename(columns = {'Distination-IP':'Destination-IP','Distination-Port':'Destination-Port'}, inplace = True)
DATA = data

# Pre-processing
print("\nPreprocessing Data...")
def convertDateDisplay(row):
  return datetime.strptime(row[1] + " " + row[2], "%m/%d/%Y %H:%M:%S")

def convertDate(row):
  #convert the date to a datetime object and then retreive the seconds since epoch
  return datetime.strptime(row[1] + " " + row[2], "%m/%d/%Y %H:%M:%S").timestamp()

def convertDuration(row):
  # Split the time by colon then zip with respective seconds
  # (3600 for hours, 60 for minutes, 1 for seconds)
  # then sume the multiples of the pairs
  return sum([x*int(t) for x,t in zip([3600,60,1],row[3].split(':'))])

# Convert date and time to seconds since epoch
data['Timestamp'] = data.apply(convertDate, axis=1)
data['Duration Seconds'] = data.apply(convertDuration, axis=1)

# Drop the No columns as it is redundant
# Drop Time, Date and Duration as theyre no longer needed
data = data.drop(['No','Start-Date','Start-Time','Duration'], axis=1)

# encode the labels for service
service_le = preprocessing.LabelEncoder()
data['Service'] = service_le.fit_transform(data['Service'])

# encode the labels for Source-IP
source_le = preprocessing.LabelEncoder()
data['Source-IP'] = source_le.fit_transform(data['Source-IP'])

# encode the labels for Destination-IP
destination_le = preprocessing.LabelEncoder()
data['Destination-IP'] = destination_le.fit_transform(data['Destination-IP'])

# Replace the dashes in destination IP and source IP
data.loc[(data['Source-Port'] == '-'),'Source-Port']= -1
data.loc[(data['Destination-Port'] == '-'),'Destination-Port']= -1

# Task 1.1
# train the model
# samples of 80000 were picked in order to produce the lowest number of anomolies detected
X_iso_forest = data
y_iso_forest = IsolationForest(max_samples=40000).fit_predict(X_iso_forest)

# Add prediction to the DATA df
DATA['Isolation Forest'] = y_iso_forest

## create Clusters of requests, changed the minimum cluster size to end with just 2 clusters.

min_size = 1
n_clusters = 0
while n_clusters != 2:
  clusters = []
  current_cluster = []
  cluster = False
  for i in range(len(y_iso_forest)):
    if y_iso_forest[i] == -1:
      cluster = True
      current_cluster.append(i)
    else:
      cluster = False
      if len(current_cluster) > min_size:
        clusters.append(current_cluster)
      current_cluster = []
  n_clusters = len(clusters)
  min_size += 1

# gathers some additional information about the clusters for exploratory purposes
def getCluster(cluster):
  protocols = {}
  ips = {}
  for event in cluster:
    protocol = DATA.iloc[event]["Service"]
    ip = DATA.iloc[event]["Source-IP"]
    if protocol in protocols:
      protocols[protocol] = protocols[protocol] + 1
    else:
      protocols[protocol] = 1
    if ip in ips:
      ips[ip] = ips[ip] + 1
    else:
      ips[ip] = 1
  cluster = {
      "idx": cluster,
      "ips": ips,
      "protocols": protocols
  }
  return cluster
cluster_one = getCluster(clusters[0])
cluster_two = getCluster(clusters[1])

print("Attack 1")
print("Start:")
print(DATA.iloc[cluster_one['idx'][0]]["Start-Date"])
print(DATA.iloc[cluster_one['idx'][0]]["Start-Time"])
print("\nEnd:")
print(DATA.iloc[cluster_one['idx'][-1]]["Start-Date"])
print(DATA.iloc[cluster_one['idx'][-1]]["Start-Time"])
print("\nServices:")
print(cluster_one['protocols'])
print("\n\nAttack 2")
print("Start:")
print(DATA.iloc[cluster_two['idx'][0]]["Start-Date"])
print(DATA.iloc[cluster_two['idx'][0]]["Start-Time"])
print("\nEnd:")
print(DATA.iloc[cluster_two['idx'][-1]]["Start-Date"])
print(DATA.iloc[cluster_two['idx'][-1]]["Start-Time"])
print("\nServices:")
print(cluster_two['protocols'])
print("\nRequests:")
print(len(cluster_two['idx']))
