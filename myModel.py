import imp
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
import myConfig as mcf
from torch.autograd import Variable
from mobilenet_v3 import MobileNetV3
from googlenet import GoogLeNet
from stacking_simple import mystacking
import copy as cp

class DNN:
    def __init__(self, net_name, args = None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.classes = args["classes"] if args["classes"] != None else None
        self.num_classes = len(self.classes) if self.classes != None else 10
        self.hyperargs = args['hyperargs'] if args['hyperargs'] == None else None
        # self.data_store = DataStore()
        self.model = self.determine_net(net_name)
        self.args = args
    def determine_net(self, net_name, pretrained=False):
        if net_name == "resnet50":
            return models.resnet50(pretrained=pretrained, num_classes=self.num_classes)
        elif net_name == "densenet":
            return models.desenet121(pretrained=pretrained, num_classes=self.num_classes)
        elif net_name == "cnn":
            return CNNet()
        elif net_name == "mnist_resnet50":
            return MnistResNet()

        elif net_name == "googlenet":
            return googlenet(num_classes=self.num_classes, hyperargs=self.hyperargs)
        # elif net_name == "mobilenet":
        #     return models.mobilenet_v3_small(pretrained=pretrained, num_classes=self.num_classes)
        else:
            raise Exception("invalid net name")

    def train(self, loader, filename):
        self.model = self.model.to(self.device)
        
        epochs = cp.deepcopy(self.args["epochs"])
        lr = 0.001
        # lr = 0.09
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        print(f'{filename}   ~~~~    ~~~~    ~~~~')
        self.model.train()
        pbar = tqdm(range(epochs))
        epoch = 0
        final_loss = 0
        epochs_add = 0
        while epoch < epochs:
            running_loss = 0.0

            for times, data in enumerate(loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            pbar.set_postfix({'loss:':running_loss/2000})
            final_loss = running_loss/2000
            # if final_loss >= 0.045 and  epochs == epoch+1:
                # if epochs % 10 == 0:
                # epochs_add += 5
                # epochs += 5
                # pbar.total = epochs
            pbar.update(1)
            epoch += 1

        pbar.close()
        print(f'loss : {final_loss}')
        torch.save(self.model.state_dict(), filename)
        del self.model
        del optimizer
        del criterion


    def load_model(self, save_name):
        # print(save_name)
        self.model.load_state_dict(torch.load(save_name))
        # print("OK")
        self.model = self.model.to(self.device)

    def model_acc(self, test_loader):
        
        self.model.eval()
        self.model = self.model.to(self.device)
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).to(self.device)
                pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()
            return correct / len(test_loader.dataset)
    
    def test(self, loader, filename):

        self.load_model(filename)
        #model = input_model
        self.model.eval()
        print(f'~~~~    ~~~~    ~~~~   {filename}   ~~~~    ~~~~    ~~~~')
        print('\n========================= Starting Test =========================')
        #print(f'------------------- {filename[:-12]} -------------------\n')
        print(f'cuda.memory_allocated: {torch.cuda.memory_allocated()}')

        correct = 0
        # total = 0
        total = len(loader.dataset)
        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print('Accuracy of the network on the 10000 test inputs: %d %%' % (100 * correct / total))

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                c = (predicted == labels).squeeze()

                for i in range(labels.shape[0]):

                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        cl = list(range(self.num_classes)) if self.classes == None else self.classes
        
        for i in range(self.num_classes):
            print('Accuracy of %5s : %2d %%' % (cl[i], 100 * class_correct[i] / class_total[i]))

        print(f'cuda.memory_allocated: {torch.cuda.memory_allocated()}')
        print('========================= Finished Test =========================\n')

    def predict_prob(self, indata):
        self.model.eval()

        with torch.no_grad():

            inputs = indata[0].unsqueeze(0)
            labels = indata[1]
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            results = F.softmax(outputs.data, dim = 1)

            return results.cpu().numpy()

    def predict(self, sample):

        correct = False
        self.model.eval()

        with torch.no_grad():

            inputs = sample[0].unsqueeze(0)
            labels = sample[1]
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            results = F.softmax(outputs.data, dim = 1)
            _,pred = torch.max(results, 1)

        return pred.cpu()



class RF:
    def __init__(self, min_samples_leaf=5) -> None:
        self.model = RandomForestClassifier(n_estimators = 2000, random_state = 100, min_samples_leaf = min_samples_leaf)
            # n_estimators : influences model effect, as larger as better but cost more
            # random_state : control tree generation in forest, each tree is different
            # bootstrap : default True, (won't be False in usual) 
            #   random sampling (put back) to generate different data set

    def train_model(self, train_x, train_y, filename = None):
        self.model.fit(train_x, train_y)
        if filename != None:
           joblib.dump(self.model, mcf.ATTACK_MODEL_PATH + filename)

    def predict_proba(self, test):
        return self.model.predict_proba(test)

    def predict(self, test_x):
            return self.model.predict(test_x)    

    def load_model(self, filename):
        return joblib.load(mcf.ATTACK_MODEL_PATH + filename)

    def model_acc(self, test_x, test_y):
        predict = self.model.predict(test_x)

        return accuracy_score(test_y, predict)
    
    def model_auc(self, test_x, test_y):
        predict = self.model.predict(test_x)
        '''
        predict_proba(x) : predict x probability of each class
          return ndarray of shape (n_samples, n_classes), or a list of such arrays
          The class probabilities of the input samples. The order of the classes corresponds to that in the attribute classes_.
        '''
        return roc_auc_score(test_y, predict)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class DT:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_leaf_nodes=15, random_state=100, min_samples_leaf=1)

    def train_model(self, train_x, train_y, filename=None):
        self.model.fit(train_x, train_y)
        if filename is not None:
            joblib.dump(self.model, mcf.ATTACK_MODEL_PATH + filename, compress=9)

    def load_model(self, filename):
        self.model = joblib.load(mcf.ATTACK_MODEL_PATH + filename)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def predict(self, test_x):
            return self.model.predict(test_x)

    def model_acc(self, test_x, test_y):
        predict = self.model.predict(test_x)
        return accuracy_score(test_y, predict)

    def model_auc(self, test_x, test_y):
        predict = self.model.predict(test_x)
        return roc_auc_score(test_y, predict)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])


class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output
        # return output, x    # return x for visualization

class mobilenet:
    def __init__(self, pretrained = False, args = None, model_mode = None, hyperargs = None) -> None:
        # super(mobilenet, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.classes = args["classes"] if args["classes"] != None else None
        self.num_classes = len(self.classes) if self.classes != None else 10
        self.model_mode = "LARGE" if model_mode == None else model_mode
        self.hyperargs = hyperargs if hyperargs != None else None
        self.model = MobileNetV3(self.model_mode, num_classes=self.num_classes, hyperargs = self.hyperargs )
        self.args = args
        # print(self.model)


    def train(self, loader, filename = None):
        self.model = self.model.to(self.device)
        
        epochs = cp.deepcopy(self.args["epochs"])
        lr = 0.001
        # lr = 0.09
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        final_loss = 0
        epochs_add = 0
        print(f'{filename}   ~~~~    ~~~~    ~~~~')
        self.model.train()
        pbar = tqdm(range(epochs))
        # for epoch in pbar:
        epoch = 0
        while epoch < epochs:
            running_loss = 0.0

            for times, data in enumerate(loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            pbar.set_postfix({'loss:':running_loss/2000})
            final_loss = running_loss/2000
            if epoch+1 == 100:
                    pbar.update(1)
                    break
            if final_loss >= 0.045 and  epochs == epoch+1:
                # if epochs % 10 == 0:
                
                epochs_add += 5
                epochs += 5
                pbar.total = epochs
            pbar.update(1)
            
            epoch += 1
        pbar.close()
        print(f'loss : {final_loss}')

        torch.save(self.model.state_dict(), filename)
        del self.model
        del optimizer
        del criterion

    def model_acc(self, test_loader):
        self.model.eval()
        self.model = self.model.to(self.device)
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).to(self.device)
                pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()
            return correct / len(test_loader.dataset)

    def load_model(self, save_name):
        self.model.load_state_dict(torch.load(save_name))
        self.model = self.model.to(self.device)

    def predict_prob(self, indata):
        self.model.eval()

        with torch.no_grad():

            inputs = indata[0].unsqueeze(0)
            labels = indata[1]
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            results = F.softmax(outputs.data, dim = 1)

            return results.cpu().numpy()

    def model_auc(self, loader):
        self.model.eval()
        prob_all = []
        label_all = []
        self.model = self.model.to(self.device)

        with torch.no_grad():
            for data,label in loader:
                data = data.to(self.device)
                prob = self.model(data) #表示模型的預測輸出
                prob_all.extend(prob[:,1].cpu().numpy()) #prob[:,1]返回每一行第二列的數，根據該函數的參數可知，y_score表示的較大標簽類的分數，因此就是最大索引對應的那個值，而不是最大索引值
                label_all.extend(label)
        return roc_auc_score(label_all,prob_all)

class SVM:
    def __init__(self) -> None:
        self.model = svm.SVC(kernel='rbf', C=1000, gamma='auto', probability=True)

    def train_model(self, train_x, train_y, filename=None):
        self.model.fit(train_x, train_y)
        if filename is not None:
            joblib.dump(self.model, mcf.ATTACK_MODEL_PATH + filename)

    def load_model(self, filename):
        self.model = joblib.load(mcf.ATTACK_MODEL_PATH + filename)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def predict(self, test_x):
            return self.model.predict(test_x)

    def model_acc(self, test_x, test_y):
        predict = self.model.predict(test_x)
        return accuracy_score(test_y, predict)

    def model_auc(self, test_x, test_y):
        predict = self.model.predict(test_x)
        return roc_auc_score(test_y, predict)

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        return roc_auc_score(test_y, pred_y[:, 1])

class googlenet:
    def __init__(self, num_classes, hyperargs = None) -> None:
        self.num_classes = num_classes
        self.hyperargs = hyperargs
        self.model = GoogLeNet(num_classes = self.num_classes, hyperargs=self.hyperargs)


class StackingEnsamble:
    def __init__(self, estimators) -> None:
        self.estimators = estimators
        self.model = StackingClassifier(estimators = self.estimators, final_estimator= LogisticRegression())

    def train(self, loader, filename = None):
        dataiter = iter(loader)
        data = dataiter.next()
        features, labels = data
        self.model.fit(features, labels)
        if filename is not None:
            joblib.dump(self.model, mcf.SHADOW_DEFENSE_MODEL_PATH + filename)
        
    def load_model(self, filename):
        self.model = joblib.load(filename)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def model_acc(self, test_x, test_y):
        predict = self.model.predict(test_x)
        return accuracy_score(test_y, predict)

    def test_model_auc(self, test_x, test_y):
        predict = self.model.predict(test_x)
        return roc_auc_score(test_y, predict)

class MyStacking:
    def __init__(self, estimator, args) -> None:
        self.args = args
        self.num_classes = 10 if args['class_num'] == None else args['class_num']
        self.model = mystacking(estimator, self.num_classes, args)
        self.classes = args['classes'] if args['classes'] != None else None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def train(self, loader, filename = None):
        self.model = self.model.to(self.device)

        print(f'~~~~ {filename}   ~~~~    ~~~~')

        epochs = cp.deepcopy(self.args["epochs"])
        lr = 0.001

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        final_loss = 0
        epochs_add = 0
        self.model.train()
        pbar = tqdm(range(epochs))
        # for epoch in pbar:
        epoch = 0
        pbar.update(0)
        while epoch < epochs:
            running_loss = 0.0
            for times, data in enumerate(loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            pbar.set_postfix({'loss:':running_loss/2000})
            final_loss = running_loss/2000
            # if final_loss >= 0.045 and  epochs == epoch+1:
                # if epochs % 10 == 0:
                # epochs_add += 10
                # epochs += 10
                # pbar.total = epochs
            pbar.update(1)
            epoch += 1
        pbar.close()
        print(f'loss : {final_loss}')

        torch.save(self.model.state_dict(), filename)
        del self.model
        del optimizer
        del criterion

    def model_acc(self, test_loader):
        self.model.eval()
        self.model = self.model.to(self.device)
        correct = 0

        with torch.no_grad():
            for images, labels in test_loader:

                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images).to(self.device)
                pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()
            return correct / len(test_loader.dataset)
    
    def load_model(self, save_name):
        self.model.load_state_dict(torch.load(save_name))
        self.model = self.model.to(self.device)

    def test(self, loader, filename):

        self.load_model(filename)

        self.model.eval()
        print(f'~~~~   {filename}   ~~~~    ~~~~')
        print('\n========================= Starting Test =========================')
        print(f'cuda.memory_allocated: {torch.cuda.memory_allocated()}')

        correct = 0
        # total = 0
        total = len(loader.dataset)
        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print('Accuracy of the network on the 10000 test inputs: %d %%' % (100 * correct / total))

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(labels.shape[0]):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        cl = list(range(self.num_classes)) if self.classes == None else self.classes
        
        for i in range(self.num_classes):
            print('Accuracy of %5s : %2d %%' % (cl[i], 100 * class_correct[i] / class_total[i]))

        print(f'cuda.memory_allocated: {torch.cuda.memory_allocated()}')
        print('========================= Finished Test =========================\n')

    def predict_prob(self, indata):
        self.model.eval()
        with torch.no_grad():
            inputs = indata[0].unsqueeze(0)
            labels = indata[1]
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            results = F.softmax(outputs.data, dim = 1)
            return results.cpu().numpy()

    def predict(self, sample):
        # correct = False
        self.model.eval()
        with torch.no_grad():
            inputs = sample[0].unsqueeze(0)
            labels = sample[1]
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            results = F.softmax(outputs.data, dim = 1)
            _,pred = torch.max(results, 1)
        return pred.cpu()


class MnistResNet(nn.Module):
    def __init__(self, in_channels=1):
        super(MnistResNet, self).__init__()

        # Load a pretrained resnet model from torchvision.models in Pytorch
        self.model = models.resnet50(pretrained=False)

        # Change the input layer to take Grayscale image, instead of RGB images. 
        # Hence in_channels is set as 1 or 3 respectively
        # original definition of the first layer on the ResNet class
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Change the output layer to output 10 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.model(x)