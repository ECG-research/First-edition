import Model as M
import Preprocessing as P

#hyperparameters
path = "First-edition\\"
sampling_rate = 100
num_eporch = 10

#load data
data = P.Preprocessing(path,sampling_rate)
X = data.get_data_x()
Y = data.get_data_y()
Sle = data.get_data_metadata()

#train
model = M.Proposed_model(1000,3,5)
model.fit([X[0],Sle[0]],Y[0],batch_size=32,validation_data = ([X[1],Sle[1]],Y[1]),epochs=num_eporch)