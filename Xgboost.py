import xgboost as xgb
import numpy

# read in data

dataset = list()
with open(filename, 'r') as file:
    csv_reader = reader(file)
for row in csv_reader:
	if not row:
		continue
	dataset.append(row)
file.close()

	
dtrain = xgb.DMatrix('train.csv')
dtest = xgb.DMatrix('test.csv')

# specify parameters via map
param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)