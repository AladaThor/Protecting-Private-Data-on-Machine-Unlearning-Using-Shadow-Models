import myModel
import create_data_set as cds
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
from torchsummary import summary

def index_2d(myList, v):
    for i, x in enumerate(myList):
        if v in x:
            return i, x.index(v)

class defense_model_set:
    def __init__(self, args = None) -> None:

        self.args = args
        self.sample_num = int(args['origin_sample_num']/10)
        self.sample_list = []
        self.Substitute_list = []
        self.Substitute_sample = []
        self.setinfo = None

    def generate_sample_list(self, _dataset_u):

        self.setinfo = cds.generateSet(_dataset_u, args=self.args)
        class_sample_num = int(self.sample_num/self.setinfo.class_num)
        tmpsample = self.setinfo._generate_index(2, sampleNum=self.sample_num)
        self.sample_list.append(tmpsample)

    def get_substitute_sample(self, unlearn_data_list, origin_set, other_set):
        self.otherset_info =  cds.generateSet(other_set, args=self.args)
        Substitute_list = []
        for index, term in enumerate(unlearn_data_list):
            term_label = origin_set[term][1]
            while True:

                tmpstitute = np.random.choice(self.otherset_info.data_table[term_label], 10, replace = False)
                if not any(item in tmpstitute for item in Substitute_list):
                    for item in tmpstitute:
                        Substitute_list.append(item)
                    break
        
        x = self.args['batch_size'] - (self.setinfo.data_num + len(Substitute_list) % self.args['batch_size'])
        
        while x > 0:
            tmpstitute = np.random.choice(self.otherset_info.data_num, 1, replace = False)
            if tmpstitute not in Substitute_list:
                Substitute_list.append(tmpstitute)
                x -= 1

        self.Substitute_sample = torch.utils.data.Subset(other_set,Substitute_list)
        self.Substitute_list = Substitute_list

            
    def merge_sample_list(self):
        tmp_list = []
        for index_class, term_class in enumerate(self.sample_list):
            for spl_class in term_class:
                tmp_list.append(spl_class)
        
        tmp_set = torch.utils.data.Subset(self.setinfo._dataset, tmp_list)
        return self.Substitute_sample + tmp_set
        
        



class defense_method:

    def __init__(self, estimator, args) -> None:
        self.args = args
        self.estimator = estimator
        self.unlearn_weight = args["weight"]
        self.stacking = myModel.MyStacking(estimator,args)
        

    def gene_stack(self, loader, filename):
        self.stacking.train(loader, filename)
        self.stacking = myModel.MyStacking(self.estimator, self.args)
        self.stacking.load_model(filename)

    def load_model(self, filename):
        self.stacking.load_model(filename)

    def predict_prob(self, test_data):
        proba_arr = self.stacking.predict_prob(test_data)
        return proba_arr

    # unlearn model & defense model weights
    def predict_prob2(self, test_data):

        prlst = []
        for index, term in enumerate(self.estimator):
            prlst.append(list(term.predict_prob(test_data).reshape(-1)))

        
        # unlearn model weight
        unlearn_weight = float(self.unlearn_weight)
        # defense model weight
        defense_total = float(1.0 - unlearn_weight)
        defense_weight = float(defense_total/(len(prlst)-1))
        

        proba_arr = []
        for i in range(self.args['class_num']):
            tmp = 0
            for j in range(len(prlst)-1):
                tmp += float(prlst[j][i]*defense_weight)
            res = tmp + (prlst[len(prlst)-1][i]*unlearn_weight)
            proba_arr.append(res)

        return proba_arr

    def predict(self, test_data):
        proba_arr = self.predict_prob(test_data)
        tmp = np.max(proba_arr)
        res = proba_arr.index(tmp)
        return res

    def predict2(self, test_data):
        proba_arr = self.predict_prob2(test_data)
        tmp = np.max(proba_arr)
        res = proba_arr.index(tmp)
        return res

    def model_acc(self, test_loader):
        return self.stacking.model_acc(test_loader)

    def model_acc2(self, test_dataset):

        correct = 0
        total = len(test_dataset)
        for index, term in enumerate(test_dataset):

            predict = self.predict2(term)

            if predict == term[1]:
                correct += 1
        return float(correct/total)
