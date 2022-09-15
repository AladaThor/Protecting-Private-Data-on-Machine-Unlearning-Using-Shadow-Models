import random as rd
import myModel as mm
import myConfig as mc
import myUtils as mu
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from os.path import exists
from unlearned_defense import defense_model_set
from unlearned_defense import defense_method
import copy as cp

def allUnique(x):
    seen = list()
    return not any(i in seen or seen.append(i) for i in x)

class generateSet:
    def __init__(self, _dataset, args = None):
        self._dataset = _dataset
        self.data_num = len(_dataset)
        self.data_table = self.CreatSetTable()
        self.used_table = []
        self.used_classC = []
        self.class_num = len(self.data_table)
        self.args = args
        self.args["class_num"] = self.class_num

    def split_data_set_balance(self, percentage = None):
        self.set_1 = []
        self.set_2 = []
        list_1 = []
        list_2 = []
        percent = percentage if percentage != None else 0.5
        if type(percentage) == type(1):
            percent = int(percentage/self.class_num)
        for i in range(self.class_num):
            # print(f'data_table length : {len(self.data_table[i])}')
            if type(percent) == type(1):
                helfclass1 = percent
            else:
                helfclass1 = int(len(self.data_table[i])*percent)
            # helfclass2 = int(len(self.data_table[i])*(1-percent))
            indexlist1 = np.random.choice(self.data_table[i], size=helfclass1, replace=False)
            indexlist2 = np.setdiff1d(self.data_table[i], indexlist1)
            for val in indexlist1:
                list_1.append(val)
            for val in indexlist2:
                list_2.append(val)
            
        self.set_1 = torch.utils.data.Subset(self._dataset, list_1)
        self.set_2 = torch.utils.data.Subset(self._dataset, list_2)
        return list_1, list_2

    def CreatSetTable(self):
        table = [[]]
        for i in range(self.data_num):

            if self._dataset[i][1] <= len(table)-1:
                table[self._dataset[i][1]].append(i)
                
            else:
                expandsize = self._dataset[i][1] - len(table) + 1
                for j in range(expandsize):
                    table.append([])
                table[self._dataset[i][1]].append(i)
        return table

    def check_table(self, index):
        classNum = self._dataset[index][1]
        if classNum > len(self.used_table)-1:
            expandsize = classNum - len(self.used_table) + 1
            for j in range(expandsize):
                self.used_table.append([])
            self.used_table[classNum].append(index)
            return False
        else:
            if index in self.used_table[classNum]:
                return True
            else:
                self.used_table[classNum].append(index)
                return False
            
    def _generate_index(self, unlearn_case, classNum = None, sampleNum = None):

        indexlist = []

        if unlearn_case == 0: 
        # unlearned many samples in same class
            if classNum == None:
                classNum = rd.randrange(0, self.class_num)
            if sampleNum == None:
                sampleNum = rd.randrange(2, len(self.data_table[classNum]))
            # print(f'classNum : {classNum}')
            for i in range(sampleNum):
                # print(f'{i} : {a}')
                while True:
                    rand_sample = rd.choice(self.data_table[classNum])
                    if not self.check_table(rand_sample):
                        indexlist.append(rand_sample)
                        break
        elif unlearn_case == 1:
        # unlearned many samples in different class (none average)
            if sampleNum == None:
                sampleNum = rd.randrange(2, 2400)
            for i in range(sampleNum):
                # print(f'{i} : {a}')
                while True:
                    rand_class = rd.randrange(0, self.class_num)
                    rand_sample = rd.choice(self.data_table[rand_class])
                    if not self.check_table(rand_sample):
                        indexlist.append(rand_sample)
                        break
        elif unlearn_case == 2:
        # unlearned many samples in different class (average)
            if sampleNum == None:
                sampleNum = rd.randrange(2, 2400)
            if classNum == None:
                classNum = range(self.class_num)
            sampleinclass = int(sampleNum/len(classNum))
            a = len(classNum)
            b = 0
            if len(self.used_table) < len(self.data_table):
                expandsize = len(self.data_table) - len(self.used_table)
                for j in range(expandsize):
                    self.used_table.append([])
            while b < a:
                tmp = len(self.data_table[classNum[b]]) - len(self.used_table[classNum[b]])
                if tmp == 0:
                    classNum.remove(classNum[b])
                    a -= 1
                    b -= 1
                elif tmp < sampleinclass :
                    sampleinclass = tmp
                b += 1
            counter = []
            for i in range(len(classNum)):
                counter.append([classNum[i],sampleinclass])
            
            for i in range(sampleinclass*len(classNum)):
                # print(f'{i} : {a}')
                while True:
                    rand_class = rd.randrange(0,len(classNum))
                    if counter[rand_class][1] > 0:
                        tmpclass = counter[rand_class][0] # class
                        rand_sample = rd.choice(self.data_table[tmpclass])
                        if not self.check_table(rand_sample):
                            indexlist.append(rand_sample)
                            counter[rand_class][1] -= 1
                            break
        elif unlearn_case == 3:
        # unlearned 1 class
            if classNum == None:
                indexlist = rd.choice(self.data_table)
            else :
                indexlist = self.data_table[classNum]
        elif unlearn_case == 4:
        # unlearned 2 class (It's a large effect.)
            if classNum == None:
                a = 2
                while a >= 1:
                    # rmclass = rd.choice(self.data_table)
                    rmclass = rd.randrange(0, self.class_num)
                    if rmclass not in self.used_classC:
                        self.used_classC.append(rmclass)
                        indexlist.append(self.data_table[rmclass])
                        a -= 1
                
            else :
                for i in classNum:
                    indexlist.append(self.data_table[i])
            
        return indexlist

    def _generate_unlearn(self, indexlist):
        tmpset = torch.utils.data.Subset(self._dataset,indexlist)

        tmploader = DataLoader(tmpset, batch_size=self.args["batch_size"], shuffle=True, num_workers=1, drop_last=True)
        return tmploader


class generate_case:
    def __init__(self, s_dataset, t_dataset, args = None) -> None:
        self.dataset_s = s_dataset
        self.dataset_t = t_dataset
        # self.set_name = args['set_name']
        self.args = args
        self.origin_sample_num = self.args["origin_sample_num"]
        self.origin_model_num = self.args["origin_model_num"]
        self.unlearn_sample_num = self.args["unlearn_sample_num"]
        self.unlearn_model_num = self.args["unlearn_model_num"]
        self.set_split_percentage = self.args["set_split_percentage"]
        self.mode = None

    def determine_mode(self, filename, mode = None):
        self.mode = mode if mode != None else 1
        if self.mode == 0:
            shadow_sample_set, target_sample_set = self._generate_OandU_model(filename)
        elif self.mode == 1:
            shadow_sample_set, target_sample_set = self._generate_multi_case(filename)
        else:
            raise Exception("invalid mode !!!   0 : O&U    1 : Multi")
        return shadow_sample_set, target_sample_set


    def _generate_OandU_model(self, filename):

        shadow_sample_set = {}
        for origin_model_index in range(self.origin_model_num):
            spt = generateSet(self.dataset_s, args=self.args)
            shalist, _ = spt.split_data_set_balance(self.origin_sample_num)
            gs1 = generateSet(spt.set_1,self.args)

            unlearn_case_list = {}
            
            for unlearn_model_index in range(self.unlearn_model_num):
                if unlearn_model_index %2 == 0:
                    unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(2, sampleNum=self.unlearn_sample_num)
                else:
                    unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(1, sampleNum=self.unlearn_sample_num)

            shadow_sample_set[origin_model_index] = {
                'shadow_list' : shalist,
                'unlearn_list': unlearn_case_list
            }

        target_sample_set = {}
        for origin_model_index in range(self.origin_model_num):
            spt = generateSet(self.dataset_t, args=self.args)
            tarlist, _ = spt.split_data_set_balance(self.origin_sample_num)
            gs1 = generateSet(spt.set_1,self.args)

            unlearn_case_list = {}
            
            for unlearn_model_index in range(self.unlearn_model_num):
                if unlearn_model_index %2 == 0:
                    unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(2, sampleNum=self.unlearn_sample_num)
                else:
                    unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(1, sampleNum=self.unlearn_sample_num)

            target_sample_set[origin_model_index] = {
                'target_list' : tarlist,
                'unlearn_list': unlearn_case_list
            }

        mu.DataStore().save_data(shadow_sample_set, filename[0])
        mu.DataStore().save_data(target_sample_set, filename[1])

        return shadow_sample_set, target_sample_set


    def _generate_multi_case(self, filename):
        
        shadow_sample_set = {}
        target_sample_set = {}
        for origin_model_index in range(self.origin_model_num):
            spt = generateSet(self.dataset_s, args=self.args)
            shalist, _ = spt.split_data_set_balance(self.set_split_percentage)
            gs1 = generateSet(spt.set_1,self.args)

            unlearn_case_list = {}

            # 1 sample in each class
            for i in range(gs1.class_num):
                unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(0,i,1)               
            
            # s10cn = [] # 10 sample in each class
            for i in range(gs1.class_num):
                unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(0,i,10)

            # all sample in each class   # sac1 = []
            for i in range(gs1.class_num):
                unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(3,classNum=i)

            # all sample in rd 2 class for 3
            for i in range(3):
                sac2 = []
                tmp = gs1._generate_index(4)
                for j, sublst in enumerate(tmp):
                    for k, val in enumerate(sublst):
                        sac2.append(val)
                unlearn_case_list[len(unlearn_case_list)] = sac2

            snce = []# n sample in each class of class_list
            for ss in range(5):
                a = rd.randrange(2+ss,11)
                b = np.random.choice(range(10),a, replace = False)
                snce.append(list(b))
            sample_num_list1 = [0.00334, 0.0067, 0.0134, 0.0234, 0.0532]
            for i, term in enumerate(sample_num_list1):
                unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(2,sampleNum = int(term*gs1.data_num), classNum=snce[i])


            sncl = [] # n sample in class_list
            sample_num_list2 = [0.005, 0.02, 0.025, 0.05, 0.1]
            for i in sample_num_list2:
                unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(1, sampleNum = int(i*gs1.data_num))

            shadow_sample_set[origin_model_index] = {
                'shadow_list' : shalist,
                'unlearn_list': unlearn_case_list
            }
        
        
        for origin_model_index in range(self.origin_model_num):
            spt = generateSet(self.dataset_t, args=self.args)
            tarlist, _ = spt.split_data_set_balance(self.set_split_percentage)
            gs1 = generateSet(spt.set_1, args=self.args)
            
            unlearn_case_list = {}

            # s1cn = [] # 1 sample in each class
            for i in range(gs1.class_num):
                unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(0,i,1)         

            # s10cn = [] # 10 sample in each class
            for i in range(gs1.class_num):
                unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(0,i,10)
        
            # sac1 = []   all sample in each class
            for i in range(gs1.class_num):
                unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(3,classNum=i)

            # all sample in rd 2 class for 3
            for i in range(3):
                sac2 = []
                tmp = gs1._generate_index(4)
                for j, sublst in enumerate(tmp):
                    for k, val in enumerate(sublst):
                        sac2.append(val)
                unlearn_case_list[len(unlearn_case_list)] = sac2
            
            snce = []# n sample in each class of class_list
            for ss in range(5):
                a = rd.randrange(2+ss,11)
                b = np.random.choice(range(10),a, replace = False)
                snce.append(list(b))
            sample_num_list1 = [0.00334, 0.0067, 0.0134, 0.0234, 0.0532]
            for i, term in enumerate(sample_num_list1):
                unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(2,sampleNum = int(term*gs1.data_num), classNum=snce[i])


            # sncl = [] # n sample in class_list
            sample_num_list2 = [0.005, 0.02, 0.025, 0.05, 0.1]
            for i in sample_num_list2:
                unlearn_case_list[len(unlearn_case_list)] = gs1._generate_index(1, sampleNum = int(i*gs1.data_num))

            target_sample_set[origin_model_index] = {
                'target_list' : tarlist,
                'unlearn_list': unlearn_case_list
            }

        mu.DataStore().save_data(shadow_sample_set, filename[0])
        mu.DataStore().save_data(target_sample_set, filename[1])
        return shadow_sample_set, target_sample_set


    def _generate_model(self, shadow_sample_set, target_sample_set):

        self.mode_name = 'OU' if self.mode == 0 else 'multi'
        
        self.attacker_shadow_origin_path = (mc.ATTACK + mc.SHADOW_MODEL + mc.SET_NAME +  
            mc.ORIGIN_MODEL + f'SMO_s{self.origin_sample_num}/')

        self.attacker_shadow_unlearn_path = (mc.ATTACK + mc.SHADOW_MODEL + mc.SET_NAME +
            mc.UNLEARN_MODEL + f'SMO_s{self.origin_sample_num}_{self.mode_name}/')

        self.target_origin_path = (mc.TARGET + mc.SET_NAME + mc.ORIGIN_MODEL + 
                f'TMO_s{self.origin_sample_num}/')
                
        self.target_unlearn_path = (mc.TARGET + mc.SET_NAME + mc.UNLEARN_MODEL + 
                f'TMO_s{self.origin_sample_num}_{self.mode_name}/')

        self.shadow_defense_path = (mc.ATTACK + mc.SHADOW_MODEL + mc.SET_NAME +
            mc.DEFENSE_MODEL + f'DMO_s{self.origin_sample_num}_{self.mode_name}/')

        self.target_defense_path = (mc.TARGET + mc.SET_NAME + 
            mc.DEFENSE_MODEL + f'DMO_s{self.origin_sample_num}_{self.mode_name}/')

        
        if not exists(self.attacker_shadow_origin_path):
            os.mkdir(self.attacker_shadow_origin_path)
        if not exists(self.attacker_shadow_unlearn_path):
            os.mkdir(self.attacker_shadow_unlearn_path)
        if not exists(self.target_origin_path):
            os.mkdir(self.target_origin_path)
        if not exists(self.target_unlearn_path):
            os.mkdir(self.target_unlearn_path)
        if not exists(self.shadow_defense_path):
            os.mkdir(self.shadow_defense_path)
        if not exists(self.target_defense_path):
            os.mkdir(self.target_defense_path)

        defense_args = cp.deepcopy(self.args)
        defense_args['epochs'] = int(defense_args['epochs']/2)


        # shadow model create
        for i, subdict in shadow_sample_set.items():

            shadow_list = subdict['shadow_list']
            unlearn_list = subdict['unlearn_list']

            originset = torch.utils.data.Subset(self.dataset_s, shadow_list)
            print(f'origin set length : {len(originset)}')
            originloader = torch.utils.data.DataLoader(originset, batch_size=32, shuffle=True, num_workers=2)
            orimodel = mm.DNN(self.args["net_name"], args = self.args)
            if not os.path.exists(self.attacker_shadow_origin_path + f'sl{i}_v1.pt'):
                orimodel.train(originloader, self.attacker_shadow_origin_path + f'sl{i}_v1.pt')
            set_list_q = list(range(len(originset)))
            for p, unlearn_sub_list in unlearn_list.items():
                print(f'unlearn_sub_list {p} : {unlearn_sub_list}')
                tmplist = np.setdiff1d(set_list_q, unlearn_sub_list)
                
                tmpset = torch.utils.data.Subset(originset,tmplist)
                print(f'unlearn set length : {len(tmpset)}')
                tmploader = torch.utils.data.DataLoader(tmpset, batch_size=32, shuffle=True, num_workers=2)
                tmpmodel = mm.DNN(self.args["net_name"], args = self.args)
                if not os.path.exists(self.attacker_shadow_unlearn_path + f'sl{i}_u{p}_v1.pt'):
                    tmpmodel.train(tmploader, self.attacker_shadow_unlearn_path + f'sl{i}_u{p}_v1.pt')
                for j in range(5):
                    defensegener = defense_model_set(args = defense_args)
                    defensegener.generate_sample_list(self.dataset_t)
                    defensegener.get_substitute_sample(unlearn_sub_list, originset, self.dataset_t)
                    defender_set = defensegener.merge_sample_list()
                    defender_loader = torch.utils.data.DataLoader(defender_set, batch_size=32, shuffle=True, num_workers=2)
                    defense_model = mm.DNN(self.args["net_name"], args = defense_args)
                    if not exists(self.shadow_defense_path + f'sl{i}_d{p}s{j}_v1.pt'):
                        defense_model.train(defender_loader, self.shadow_defense_path + f'sl{i}_d{p}s{j}_v1.pt')

    # target model create
        for i, subdict in target_sample_set.items():
            target_list = subdict['target_list']
            unlearn_list = subdict['unlearn_list']
            originset = torch.utils.data.Subset(self.dataset_t, target_list)
            originloader = torch.utils.data.DataLoader(originset, batch_size=32, shuffle=True, num_workers=2)
            orimodel = mm.DNN(self.args["net_name"], args = self.args)
            if not os.path.exists(self.target_origin_path + f'sl{i}_v1.pt'):
                    orimodel.train(originloader, self.target_origin_path + f'sl{i}_v1.pt')
            set_list_q = list(range(len(originset)))
            for p, unlearn_sub_list in unlearn_list.items():
                tmplist = np.setdiff1d(set_list_q, unlearn_sub_list)
                unlearnset = torch.utils.data.Subset(originset,tmplist)
                unlearnloader = torch.utils.data.DataLoader(unlearnset, batch_size=32, shuffle=True, num_workers=2)
                unlearnmodel = mm.DNN(self.args["net_name"], args = self.args)
                if not os.path.exists(self.target_unlearn_path + f'sl{i}_u{p}_v1.pt'):
                    unlearnmodel.train(unlearnloader, self.target_unlearn_path + f'sl{i}_u{p}_v1.pt')
                # Defense model
                for j in range(5):
                    defensegener = defense_model_set(args = defense_args)
                    defensegener.generate_sample_list(self.dataset_s)
                    defensegener.get_substitute_sample(unlearn_sub_list, originset, self.dataset_s)
                    defender_set = defensegener.merge_sample_list()
                    defender_loader = torch.utils.data.DataLoader(defender_set, batch_size=32, shuffle=True, num_workers=2)
                    defense_model = mm.DNN(self.args["net_name"], args = defense_args)
                    if not exists(self.target_defense_path + f'sl{i}_d{p}s{j}_v1.pt'):
                        defense_model.train(defender_loader, self.target_defense_path + f'sl{i}_d{p}s{j}_v1.pt')


    def get_model_predict(self, shadow_sample_set, target_sample_set):

        weight_str = int(self.args['weight']*100)
        self.mode_name = 'OU' if self.mode == 0 else 'multi'

        self.attacker_shadow_origin_path = (mc.ATTACK + mc.SHADOW_MODEL + mc.SET_NAME +  
            mc.ORIGIN_MODEL + f'SMO_s{self.origin_sample_num}/')

        self.attacker_shadow_unlearn_path = (mc.ATTACK + mc.SHADOW_MODEL + mc.SET_NAME +
            mc.UNLEARN_MODEL + f'SMO_s{self.origin_sample_num}_{self.mode_name}/')

        self.target_origin_path = (mc.TARGET + mc.SET_NAME + mc.ORIGIN_MODEL + 
                f'TMO_s{self.origin_sample_num}/')
                
        self.target_unlearn_path = (mc.TARGET + mc.SET_NAME + mc.UNLEARN_MODEL + 
                f'TMO_s{self.origin_sample_num}_{self.mode_name}/')

        self.shadow_defense_path = (mc.ATTACK + mc.SHADOW_MODEL + mc.SET_NAME +
            mc.DEFENSE_MODEL + f'DMO_s{self.origin_sample_num}_{self.mode_name}/')

        self.target_defense_path = (mc.TARGET + mc.SET_NAME + 
            mc.DEFENSE_MODEL + f'DMO_s{self.origin_sample_num}_{self.mode_name}/')

        shadow_predict_df = pd.DataFrame(columns=["origin", "unlearn", "label"])
        shadef_predict_df1 = pd.DataFrame(columns=["origin", "unlearn", "label"])
        shadef_predict_df3 = pd.DataFrame(columns=["origin", "unlearn", "label"])
        shadef_predict_df5 = pd.DataFrame(columns=["origin", "unlearn", "label"])
        
        for shadow_set_index, shadow_set_case in shadow_sample_set.items():
            shadow_list = shadow_set_case["shadow_list"]
            unlearn_list = shadow_set_case["unlearn_list"]

            originset = torch.utils.data.Subset(self.dataset_s, shadow_list)
            set_list_q = list(range(len(originset)))
            origin_model = mm.DNN(net_name = mc.NET_NAME, args = self.args)
            origin_model.load_model(self.attacker_shadow_origin_path + f'sl{shadow_set_index}_v1.pt')
            
            for unlearn_term_index, unlearn_list_term in unlearn_list.items():

                unlearn_model = mm.DNN(net_name = mc.NET_NAME, args = self.args)
                unlearn_model.load_model(self.attacker_shadow_unlearn_path +
                    f'sl{shadow_set_index}_u{unlearn_term_index}_v1.pt')
                
                ds_tmp1 = []
                ds_tmp3 = []
                ds_tmp5 = []
                for j in range(5):
                    defense_model = mm.DNN(net_name=self.args["net_name"], args=self.args)
                    defense_model.load_model(self.shadow_defense_path + 
                        f'sl{shadow_set_index}_d{unlearn_term_index}s{j}_v1.pt')
                    ds_tmp5.append(defense_model)

                ds_tmp1 = ds_tmp5[:1]
                ds_tmp3 = ds_tmp5[:3]

                ds_tmp1.append(unlearn_model)
                ds_tmp3.append(unlearn_model)
                ds_tmp5.append(unlearn_model)

                defender1 = defense_method(ds_tmp1, args=self.args)
                defender3 = defense_method(ds_tmp3, args=self.args)
                defender5 = defense_method(ds_tmp5, args=self.args)

                tmpdf = pd.DataFrame(columns=["origin", "unlearn", "label"])
                dtmpdf1 = pd.DataFrame(columns=["origin", "unlearn", "label"])
                dtmpdf3 = pd.DataFrame(columns=["origin", "unlearn", "label"])
                dtmpdf5 = pd.DataFrame(columns=["origin", "unlearn", "label"])

                for unlearn_sample_index, unlearn_sample in enumerate(unlearn_list_term):
                    
                    test_case = originset[unlearn_sample]
                    predict_origin = origin_model.predict_prob(test_case)
                    predict_unlearn = unlearn_model.predict_prob(test_case)
                    predict_defense1 = defender1.predict_prob2(test_case)
                    predict_defense3 = defender3.predict_prob2(test_case)
                    predict_defense5 = defender5.predict_prob2(test_case)

                    tmpdf.loc[len(tmpdf)] = [predict_origin, predict_unlearn, 1]
                    dtmpdf1.loc[len(dtmpdf1)] = [predict_origin, predict_defense1, 1]
                    dtmpdf3.loc[len(dtmpdf3)] = [predict_origin, predict_defense3, 1]
                    dtmpdf5.loc[len(dtmpdf5)] = [predict_origin, predict_defense5, 1]


                other_sample_term = np.setdiff1d(set_list_q, unlearn_list_term)
                other_sample_list = np.random.choice(other_sample_term, size=len(unlearn_list_term), replace=False)
                
                for i, other_sample in enumerate(other_sample_list):
                    test_case = originset[other_sample]
                    predict_origin = origin_model.predict_prob(test_case)
                    predict_unlearn = unlearn_model.predict_prob(test_case)
                    predict_defense1 = defender1.predict_prob2(test_case)
                    predict_defense3 = defender3.predict_prob2(test_case)
                    predict_defense5 = defender5.predict_prob2(test_case)

                    tmpdf.loc[len(tmpdf)] = [predict_origin, predict_unlearn, 0]
                    dtmpdf1.loc[len(dtmpdf1)] = [predict_origin, predict_defense1, 0]
                    dtmpdf3.loc[len(dtmpdf3)] = [predict_origin, predict_defense3, 0]
                    dtmpdf5.loc[len(dtmpdf5)] = [predict_origin, predict_defense5, 0]


                shadow_predict_df = pd.concat([shadow_predict_df,tmpdf], axis=0, ignore_index=True)
                shadef_predict_df1 = pd.concat([shadef_predict_df1, dtmpdf1], axis=0, ignore_index=True)
                shadef_predict_df3 = pd.concat([shadef_predict_df3, dtmpdf3], axis=0, ignore_index=True)
                shadef_predict_df5 = pd.concat([shadef_predict_df5, dtmpdf5], axis=0, ignore_index=True)

        mu.DataStore().save_data(shadow_predict_df,
            mc.ATTACK + mc.SET_NAME + f'/SMO_S{self.origin_sample_num}_{self.mode_name}_predict_set_v1.pickle')
        mu.DataStore().save_data(shadef_predict_df1,
            mc.ATTACK + mc.SET_NAME + f'/SMO_S{self.origin_sample_num}_{self.mode_name}_predict_set_{weight_str}_defense1_v1.pickle')
        mu.DataStore().save_data(shadef_predict_df3,
            mc.ATTACK + mc.SET_NAME + f'/SMO_S{self.origin_sample_num}_{self.mode_name}_predict_set_{weight_str}_defense3_v1.pickle')    
        mu.DataStore().save_data(shadef_predict_df5,
            mc.ATTACK + mc.SET_NAME + f'/SMO_S{self.origin_sample_num}_{self.mode_name}_predict_set_{weight_str}_defense5_v1.pickle')               

        target_predict_df = pd.DataFrame(columns=["origin", "unlearn", "label"])
        defense_predict_df1 = pd.DataFrame(columns=["origin", "unlearn", "label"])
        defense_predict_df3 = pd.DataFrame(columns=["origin", "unlearn", "label"])
        defense_predict_df5 = pd.DataFrame(columns=["origin", "unlearn", "label"])

        for target_set_index, target_set_case in target_sample_set.items():
            
            target_list = target_set_case["target_list"]
            unlearn_list = target_set_case["unlearn_list"]

            originset = torch.utils.data.Subset(self.dataset_t, target_list)
            set_list_q = list(range(len(originset)))
            origin_model = mm.DNN(net_name=self.args["net_name"], args=self.args)
            origin_model.load_model(self.target_origin_path + f'sl{target_set_index}_v1.pt')

            for unlearn_term_index, unlearn_list_term in unlearn_list.items():

                unlearn_model = mm.DNN(net_name=self.args["net_name"], args=self.args)
                unlearn_model.load_model(self.target_unlearn_path +
                    f'sl{target_set_index}_u{unlearn_term_index}_v1.pt')


                ds_tmp1 = []
                ds_tmp3 = []
                ds_tmp5 = []
                for j in range(5):
                    defense_model = mm.DNN(net_name=self.args["net_name"], args=self.args)
                    defense_model.load_model(self.target_defense_path + 
                        f'sl{target_set_index}_d{unlearn_term_index}s{j}_v1.pt')
                    ds_tmp5.append(defense_model)

                ds_tmp1 = ds_tmp5[:1]
                ds_tmp3 = ds_tmp5[:3]

                ds_tmp1.append(unlearn_model)
                ds_tmp3.append(unlearn_model)
                ds_tmp5.append(unlearn_model)

                tmplist = np.setdiff1d(set_list_q, unlearn_list_term)
                unlearnset = torch.utils.data.Subset(originset,tmplist)
                unlearnloader = torch.utils.data.DataLoader(unlearnset, batch_size=32, shuffle=True, num_workers=2)

                tmpdf = pd.DataFrame(columns=["origin", "unlearn", "label"])
                dtmpdf1 = pd.DataFrame(columns=["origin", "unlearn", "label"])
                dtmpdf3 = pd.DataFrame(columns=["origin", "unlearn", "label"])
                dtmpdf5 = pd.DataFrame(columns=["origin", "unlearn", "label"])

                defender1 = defense_method(ds_tmp1, args=self.args)
                defender3 = defense_method(ds_tmp3, args=self.args)
                defender5 = defense_method(ds_tmp5, args=self.args)

                for unlearn_sample_index, unlearn_sample in enumerate(unlearn_list_term):
                    
                    test_case = originset[unlearn_sample]
                    predict_origin = origin_model.predict_prob(test_case)
                    predict_unlearn = unlearn_model.predict_prob(test_case)
                    predict_defense1 = defender1.predict_prob2(test_case)
                    predict_defense3 = defender3.predict_prob2(test_case)
                    predict_defense5 = defender5.predict_prob2(test_case)

                    tmpdf.loc[len(tmpdf)] = [predict_origin, predict_unlearn, 1]
                    dtmpdf1.loc[len(dtmpdf1)] = [predict_origin, predict_defense1, 1]
                    dtmpdf3.loc[len(dtmpdf3)] = [predict_origin, predict_defense3, 1]
                    dtmpdf5.loc[len(dtmpdf5)] = [predict_origin, predict_defense5, 1]

                other_sample_term = np.setdiff1d(set_list_q, unlearn_list_term)
                other_sample_list = np.random.choice(other_sample_term, size=len(unlearn_list_term), replace=False)
                
                for i, other_sample in enumerate(other_sample_list):
                    test_case = originset[other_sample]
                    predict_origin = origin_model.predict_prob(test_case)
                    predict_unlearn = unlearn_model.predict_prob(test_case)
                    predict_defense1 = defender1.predict_prob2(test_case)
                    predict_defense3 = defender3.predict_prob2(test_case)
                    predict_defense5 = defender5.predict_prob2(test_case)

                    tmpdf.loc[len(tmpdf)] = [predict_origin, predict_unlearn, 0]
                    dtmpdf1.loc[len(dtmpdf1)] = [predict_origin, predict_defense1, 0]
                    dtmpdf3.loc[len(dtmpdf3)] = [predict_origin, predict_defense3, 0]
                    dtmpdf5.loc[len(dtmpdf5)] = [predict_origin, predict_defense5, 0]

                target_predict_df = pd.concat([target_predict_df,tmpdf], axis=0, ignore_index=True)
                defense_predict_df1 = pd.concat([defense_predict_df1,dtmpdf1], axis=0, ignore_index=True)
                defense_predict_df3 = pd.concat([defense_predict_df3,dtmpdf3], axis=0, ignore_index=True)
                defense_predict_df5 = pd.concat([defense_predict_df5,dtmpdf5], axis=0, ignore_index=True)

        mu.DataStore().save_data(target_predict_df,
            mc.TARGET + mc.SET_NAME + f'/TMO_S{self.origin_sample_num}_{self.mode_name}_predict_set_v1.pickle')           
        mu.DataStore().save_data(defense_predict_df1,
            mc.TARGET + mc.SET_NAME + f'/TMO_S{self.origin_sample_num}_{self.mode_name}_predict_set_defense1_v1.pickle')
        mu.DataStore().save_data(defense_predict_df3,
            mc.TARGET + mc.SET_NAME + f'/TMO_S{self.origin_sample_num}_{self.mode_name}_predict_set_defense3_v1.pickle')  
        mu.DataStore().save_data(defense_predict_df5,
            mc.TARGET + mc.SET_NAME + f'/TMO_S{self.origin_sample_num}_{self.mode_name}_predict_set_defense5_v1.pickle')     




    def reshape_for_attack(self, predict_df):
        tmp_predict_df = pd.DataFrame(columns=["data", "label"])
        for i in range(len(predict_df)):
            tmpo = np.array(predict_df.origin[i], dtype = np.float32)
            tmpu = np.array(predict_df.unlearn[i], dtype = np.float32)
            valtmp = np.append(tmpo, tmpu)
            
            tmp_predict_df.loc[i] = [valtmp, predict_df.label[i]]
        return tmp_predict_df


    def obtain_feature(self, post_df):
        tpost_df = cp.deepcopy(post_df)
        posterior_df = {}
        
        # direct_diff
        valda = []
        for index, posterior in enumerate(tpost_df.origin):
            valda.append(np.array(tpost_df.origin[index]).reshape(-1) - np.array(tpost_df.unlearn[index]).reshape(-1))
        
        posterior_df["direct_diff"] = {
            'data' : np.array(valda),
            'label' : tpost_df.label
        }
        
        # sorted_diff
        valda = []
        for index, posterior in enumerate(tpost_df.origin):
            sort_indices = np.argsort(posterior[0, :])
            tmpunlearn = np.array(tpost_df.unlearn[index]).reshape(1,-1)

            tpost_df.origin[index] = posterior[0, sort_indices].reshape((1, sort_indices.size))
            tpost_df.unlearn[index] = tmpunlearn[0, sort_indices].reshape((1, sort_indices.size))
            tmp = tpost_df.origin[index] - tpost_df.unlearn[index]

            valda.append(tmp)

        valda = np.array(valda).squeeze()

        posterior_df["sorted_diff"] = {
            'data' : valda,
            'label' : tpost_df.label
        }

        # l2_distance
        from scipy.spatial import distance
        valda = []
        for index in range(tpost_df.shape[0]):
            original_posterior = tpost_df.origin[index][0]
            unlearning_posterior = tpost_df.unlearn[index][0]
            euclidean = distance.euclidean(original_posterior, unlearning_posterior)
            valda.append(np.full((1, 1), euclidean).reshape(1))
        posterior_df["l2_distance"] = {
            'data' : np.array(valda),
            'label' : tpost_df.label
        }

        # direct_concat
        valda = []
        for index in range(tpost_df.shape[0]):
            original_posterior = tpost_df.origin[index]
            unlearning_posterior = tpost_df.unlearn[index]
            conc = np.concatenate((original_posterior, unlearning_posterior), axis=1)
            conc = conc.reshape(20)
            valda.append(conc)
        posterior_df["direct_concat"] = {
            'data' : np.array(valda),
            'label' : tpost_df.label
        }

        # sorted_concat
        valda = []
        for index, posterior in enumerate(tpost_df.origin):
            sort_indices = np.argsort(posterior[0, :])
            original_posterior = posterior[0, sort_indices].reshape((1, sort_indices.size))
            unlearning_posterior = tpost_df.unlearn[index][0, sort_indices].reshape((1, sort_indices.size))
            conc = np.concatenate((original_posterior, unlearning_posterior), axis=1)
            conc = conc.reshape(20)
            valda.append(conc)
        posterior_df["sorted_concat"] = {
            'data' : np.array(valda),
            'label' : tpost_df.label
        }

        return posterior_df