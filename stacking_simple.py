# import os
# from sklearn.metrics import accuracy_score, roc_auc_score
# from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy as cp
# import torch.optim as optim
# import torchvision.models as models
# from sklearn.ensemble import RandomForestClassifier, StackingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import svm
# from sklearn.linear_model import LogisticRegression
# import joblib
# import numpy as np
# import myConfig as mcf
# import myModel as mm 

class mystacking(nn.Module):
    def __init__(self, estimator, num_classes, args) -> None:
        super(mystacking, self).__init__()
        self.estnum = len(estimator)
        self.models = []
        for i in range(self.estnum-1):
            self.models.append(cp.deepcopy(estimator[i].model))
        # self.model_0 = cp.deepcopy(estimator[0].model)
        # self.model_1 = cp.deepcopy(estimator[1].model)
        # self.model_2 = cp.deepcopy(estimator[2].model)
        # self.model_3 = cp.deepcopy(estimator[3].model)
        # self.model_4 = cp.deepcopy(estimator[4].model)
        self.heavymodel = cp.deepcopy(estimator[len(estimator)-1].model)

        for submodel in self.models:
            submodel.fc = nn.Identity()
        # self.model_0.fc = nn.Identity()
        # self.model_1.fc = nn.Identity()
        # self.model_2.fc = nn.Identity()
        # self.model_3.fc = nn.Identity()
        # self.model_4.fc = nn.Identity()
        self.heavymodel.fc = nn.Identity()

        for submodel in self.models:
            for param in submodel.parameters():
                param.requires_grad_(False)
        # for param in self.model_0.parameters():
        #     param.requires_grad_(False)
        # for param in self.model_1.parameters():
        #     param.requires_grad_(False)
        # for param in self.model_2.parameters():
        #     param.requires_grad_(False)
        # for param in self.model_3.parameters():
        #     param.requires_grad_(False)
        # for param in self.model_4.parameters():
        #     param.requires_grad_(False)
        for param in self.heavymodel.parameters():
            param.requires_grad_(False)

        self.linkfc = nn.Linear(2048*self.estnum-1, 2048)
        self.classifier = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            x_base = []
            for submodel in self.models:
                tmp = submodel(x.clone())
                tmp = tmp.view(tmp.size(0), -1)
                x_base.append(tmp)

            # x0 = self.model_0(x.clone())  # clone to make sure x is not changed by inplace methods
            # x0 = x0.view(x0.size(0), -1)
            # x1 = self.model_1(x.clone())
            # x1 = x1.view(x1.size(0), -1)
            # x2 = self.model_2(x.clone())
            # x2 = x2.view(x2.size(0), -1)
            # x3 = self.model_3(x.clone())
            # x3 = x3.view(x3.size(0), -1)
            # x4 = self.model_4(x.clone())
            # x4 = x4.view(x4.size(0), -1)
            x_heavy = self.heavymodel(x.clone())
            x_heavy = x_heavy.view(x_heavy.size(0), -1)

        # x_base = torch.cat((x0, x1, x2, x3, x4), dim=1)
        tmp = x_base[0]
        for subx in range(1,len(x_base)):
            tmp = torch.cat((x_base[subx], x_heavy), dim=1)
        x_base = torch.cat((tmp, x_heavy), dim=1)
        x_base = self.linkfc(F.relu(x_base))

        
        # x = torch.cat((x_base, x_heavy), dim=1)
        x = self.classifier(F.relu(x_base))
        return x
