from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.backends import cudnn
import os
import matplotlib.pyplot as plt
import numpy as np


# Hyperparameters
num_epochs = 100
num_classes = 2
batch_size = 10000
learning_rate = 0.001
result_path = './logs'

# Importing csv dataset:

# total_df = pd.read_csv('./Chicago_dataset/total_df.csv')
# X_all = total_df[['Beat', 'District', 'Latitude', 'Longitude', 'Month', 'Weekday', 'Block_Num', 'Block', 'Block_Street',
#                  'Primary_Type', 'Description', 'Location_Description', 'Domestic', 'Year']]
# y_all = total_df[['Arrest']]

X_all_array = np.loadtxt('./Chicago_dataset/X_all_array.csv', delimiter=',')
y_all_array = np.loadtxt('./Chicago_dataset/y_all_array.csv', delimiter=',')

# # Sampling skew using numpy:
# X_all_array1 = X_all.to_numpy()
# y_all_array1 = y_all.to_numpy()
# neg_count=0
# j=0
# X_all_array = np.zeros((3624386,14))
# y_all_array = np.zeros((3624386,1))
# for i in range(0,len(X_all_array1)):
#     # if i%100000 == 0:
#     #     print(i)
#     if y_all_array1[i] == 1:
#         X_all_array[j][:] = X_all_array1[i][:]
#         y_all_array[j][:] = y_all_array1[i][:]
#         j+=1
#     elif y_all_array1[i] == 0 and neg_count<1900000:
#         neg_count+=1
#         X_all_array[j][:] = X_all_array1[i][:]
#         y_all_array[j][:] = y_all_array1[i][:]
#         j+=1
#
# neg=0
# pos=0
# for i in range(0,len(X_all_array)):
#     if y_all_array[i]==0:
#         neg+=1
#     elif y_all_array[i] == 1:
#         pos +=1
#     else:
#         print("Problem")
#
# np.savetxt('./Chicago_dataset/X_all_array.csv', X_all_array, delimiter=',')
# np.savetxt('./Chicago_dataset/y_all_array.csv', y_all_array, delimiter=',')


# Applying five-fold cross validation split:
X_train, X_test, y_train, y_test = train_test_split(X_all_array, y_all_array, random_state=1, test_size=0.2)
# X_train1, X_test1, y_train1, y_test1 = train_test_split(X_all_array, y_all_array, random_state=1, test_size=0.01)
# X_train, X_test, y_train, y_test = train_test_split(X_test1, y_test1, random_state=1, test_size=0.2)

# Testing data skew:
X_train_array1 = X_train
y_train_array1 = y_train.astype(int)
X_test_array1 = X_test
y_test_array1 = y_test.astype(int)
pos=0
neg=0
for i in range(0,len(X_train_array1)):
    if y_train_array1[i] == 1:
        pos+=1
    elif y_train_array1[i] == 0:
        neg += 1
    else:
        print('problem')

# Normalizing the data:
X_train_array = X_train_array1 - np.mean(X_train_array1,axis=0)
X_train_array = X_train_array/ np.std(X_train_array,axis=0)

X_test_array = X_test_array1 - np.mean(X_test_array1,axis=0)
X_test_array = X_test_array/ np.std(X_test_array,axis=0)

# my_x = [np.array([[1.0,2],[3,4]]),np.array([[5.,6],[7,8]])] # a list of numpy arrays
# my_y = [np.array([4.]), np.array([2.])] # another list of numpy arrays (targets)
# tensor_x_temp = torch.Tensor(my_x)# transform to torch tensor

# Creating dataset and dataloader:
tensor_x_train = torch.Tensor(X_train_array)
tensor_y_train = torch.tensor(y_train_array1,dtype=torch.long)
tensor_x_test = torch.Tensor(X_test_array)
tensor_y_test = torch.Tensor(y_test_array1)

dataset_train = TensorDataset(tensor_x_train,tensor_y_train) # create your dataset
dataset_test = TensorDataset(tensor_x_test,tensor_y_test)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size) # create your dataloader
dataloader_test = DataLoader(dataset_test,batch_size=batch_size)

# train_labels = dataset_train.labels
# test_labels = dataset_test.labels

# network architecture:
class LinNet(nn.Module):
    def __init__(self):
        super(LinNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(14, 28),
            # nn.BatchNorm1d(28),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(28, 112),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(112, 448),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(448, 448),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(448, 448),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Linear(448, 448),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Linear(448, 224),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Linear(224, 112),
            nn.ReLU()
        )
        self.layer9 = nn.Sequential(
            nn.Linear(112, 56),
            nn.ReLU()
        )
        self.layer10 = nn.Sequential(
            nn.Linear(56, 28),
            nn.ReLU()
        )
        self.layer11 = nn.Sequential(
            nn.Linear(28, 14),
            nn.ReLU()
        )
        self.layer12 = nn.Sequential(
            nn.Linear(14, 2),
            #nn.ReLU()
            # nn.Softmax(2)
        )
        self.drop_out = nn.Dropout()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        return out

USE_GPU = True
model = LinNet()
print(model)
if USE_GPU:
    model.cuda()

## name of the architecture
architecture = '{}-{}'.format(model, 'train')


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# criterion = nn.BCEWithLogitsLoss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10000, gamma=0.5)


# Train the model
total_step = len(dataloader_train)
loss_list = []
acc_list = []

## Training Stage
train_pred = torch.zeros((tensor_x_train.shape[0]))
cudnn.benchmark = True
X = []
X1= []
Y = []
Y1 = []
Epoch_accuracy = []
for epoch in range(num_epochs):
    index = 0
    for i, (images, labels) in enumerate(dataloader_train):
        images = images.cuda()
        outputs = model(images)
        # if epoch == (num_epochs-1):
        #     print(epoch,'::')
        #     outputs1 = outputs.cpu().detach().numpy()
        #     X.append(outputs1)
        #     Y.append(torch.max(labels, 1)[0])
        # if epoch == 0:
        #     outputs1 = outputs.cpu().detach().numpy()
        #     X1.append(outputs1)
        #     Y1.append(torch.max(labels, 1)[0])
        # X.append(images)
        # Y.append(torch.max(labels, 1)[0])

        loss = criterion(outputs,labels.cuda())
        loss_list.append(loss.item())

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels.cuda()).sum().item()
        acc_list.append(correct / total)

        train_pred[(batch_size * index):(batch_size * index + batch_size)] = predicted
        index += 1

        if (i + 1) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

    #     ###### MODEL SAVE ######
    # if (epoch+1) % 10 == 0:
    #     save_file_path = os.path.join(result_path, 'save_{}.pth'.format(epoch))
    #     save_file_path = './model/nn2'
    #     states = {
    #         'epochs': epoch,
    #         'arch': architecture,
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #     }
    #
    #     torch.save(states, save_file_path)

######### MODEL LOAD #######
# resume_from_model = True
# model_path = './logs/save_29.pth'
# model_path = './model/nn1'
test_pred = torch.zeros((tensor_x_test.shape[0]))
# if resume_from_model:
#     print('loading checkpoint {}'.format(model_path))
#     checkpoint = torch.load(model_path)
#     assert architecture == checkpoint['arch']
#
#     begin_epoch = checkpoint['epochs']
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])
#

#Test the model
index =0
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in dataloader_test:
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.cuda().size(0)
        correct += (predicted == labels.cuda()).sum().item()

        #X.append(labels.cpu().detach().numpy())
        test_pred[(batch_size * index):(batch_size * index + batch_size)] = predicted
        index += 1


    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))

# Save the model and plot the loss and accuracy
# p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch LinNet results')
# p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
# p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
# p.line(np.arange(len(loss_list)), loss_list)
# p.line(np.arange(len(loss_list)), np.array(acc_list) * 100, y_range_name='Accuracy', color='red')
# show(p)

# Calculating accuracy parameters for the nn classifier:
sum =0
fp = 0
fn = 0
tp = 0
tn = 0
pred_final = test_pred.cpu().detach().numpy()
for i in range(0,len(pred_final)):
    sum += pred_final[i] == y_test_array1[i]
    if (pred_final[i] == 1) and (y_test_array1[i] == 1):
        tp += 1
    elif (pred_final[i] == 1) and (y_test_array1[i] == 0):
        fp += 1
    elif (pred_final[i] == 0) and (y_test_array1[i] == 1):
        fn += 1
    elif (pred_final[i] == 0) and (y_test_array1[i] == 0):
        tn += 1
    else:
        print("Fetal Error: Check accuracy and recall")



print("nn Classifier:\nAccuracy: ", sum/len(y_test_array1)*100, "%")
Precision = tp / (tp+fp)
Recall = tp / (tp+fn)
f1_score = 2*(Recall * Precision) / (Recall + Precision)
print("Precision: ", Precision, "\nRecall: ", Recall, "\nf1_score: ", f1_score)