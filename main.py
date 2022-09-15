from argment import argparser
import torch
from create_data_set import generateSet, generate_case, allUnique
import myUtils as mu
import myModel as mm
import myConfig as mc
import time
import os
from os.path import exists

args = argparser()

rain_loader, test_loader, train_set, test_set, classes = mu.LoadData.load_image(args["set_name"])
mu.DataStore().create_basic_folders()

args['classes'] = classes
args['class_num'] = len(classes)
origin_sample_num = args['origin_sample_num']
mode_name = args['mode_name']

stamp = time.localtime(time.time()) if args["data_file_path"] == "default" else args["data_file_path"]
DaCif = mc.DATASET_PATH + mc.SET_NAME
dirname = f'{stamp}_s{origin_sample_num}_v1'

if not exists(DaCif + dirname):
    os.mkdir(DaCif + dirname)

shasamset_file =  f'{DaCif}{dirname}/shadow_sample_set_dict.pickle'
tarsamset_file = f'{DaCif}{dirname}/target_sample_set_dict.pickle'
shasamlst_file =  f'{DaCif}{dirname}/shadow_origin_set_index_list.pickle'
tarsamlst_file = f'{DaCif}{dirname}/target_origin_set_index_list.pickle'
filename = [shasamset_file, tarsamset_file]

# dataset index create and backup
if not exists(shasamlst_file) and not exists(tarsamlst_file):
    STset = generateSet(train_set)
    shadow_set_list, target_set_list = STset.split_data_set_balance()
    mu.DataStore().save_data(shadow_set_list, shasamlst_file)
    mu.DataStore().save_data(target_set_list, tarsamlst_file)
else:
    shadow_origin_list = mu.DataStore().load_data(shasamlst_file)
    target_origin_list = mu.DataStore().load_data(tarsamlst_file)
    shadow_set = torch.utils.data.Subset(train_set,shadow_origin_list)
    target_set = torch.utils.data.Subset(train_set,target_origin_list)

generator = generate_case(shadow_set, target_set, args=args)

# origin model & unlearn model set index dictionary
if not exists(shasamset_file) and not exists(tarsamset_file):
    shadow_sample_set, target_sample_set = generator.determine_mode(filename,mode=args["mode"])
else:
    shadow_sample_set = mu.DataStore().load_data(shasamset_file)
    target_sample_set = mu.DataStore().load_data(tarsamset_file)

generator.mode = args["mode"]

generator._generate_model(shadow_sample_set, target_sample_set)
generator.get_model_predict(shadow_sample_set, target_sample_set)

shadow_defense0 = mu.DataStore.load_data(mc.ATTACK + mc.SET_NAME + f'/SMO_S{origin_sample_num}_{mode_name}_predict_set_v1.pickle')
shadow_defense1 = mu.DataStore.load_data(mc.ATTACK + mc.SET_NAME + f'/SMO_S{origin_sample_num}_{mode_name}_predict_set_defense1_v1.pickle')
shadow_defense3 = mu.DataStore.load_data(mc.ATTACK + mc.SET_NAME + f'/SMO_S{origin_sample_num}_{mode_name}_predict_set_defense3_v1.pickle')
shadow_defense5 = mu.DataStore.load_data(mc.ATTACK + mc.SET_NAME + f'/SMO_S{origin_sample_num}_{mode_name}_predict_set_defense5_v1.pickle')

target_defense0 = mu.DataStore.load_data(mc.TARGET + mc.SET_NAME + f'/TMO_S{origin_sample_num}_{mode_name}_predict_set_v1.pickle')
target_defense1 = mu.DataStore.load_data(mc.TARGET + mc.SET_NAME + f'/TMO_S{origin_sample_num}_{mode_name}_predict_set_defense1_v1.pickle')
target_defense3 = mu.DataStore.load_data(mc.TARGET + mc.SET_NAME + f'/TMO_S{origin_sample_num}_{mode_name}_predict_set_defense3_v1.pickle')
target_defense5 = mu.DataStore.load_data(mc.TARGET + mc.SET_NAME + f'/TMO_S{origin_sample_num}_{mode_name}_predict_set_defense5_v1.pickle')

# 0 -> no defense
# n -> defense with n shadow model

# shadow model prediction
tmp_df = generator.reshape_for_attack(shadow_defense0)
shadow_train_x0 = tmp_df.data.tolist()
shadow_train_y0 = tmp_df['label'].tolist()
train_attacker_loader0 = mu.get_dataloader(tmp_df)

tmp_df = generator.reshape_for_attack(shadow_defense1)
shadow_train_x1 = tmp_df.data.tolist()
shadow_train_y1 = tmp_df['label'].tolist()
train_attacker_loader1 = mu.get_dataloader(tmp_df)

tmp_df = generator.reshape_for_attack(shadow_defense3)
shadow_train_x3 = tmp_df.data.tolist()
shadow_train_y3 = tmp_df['label'].tolist()
train_attacker_loader3 = mu.get_dataloader(tmp_df)

tmp_df = generator.reshape_for_attack(shadow_defense5)
shadow_train_x5 = tmp_df.data.tolist()
shadow_train_y5 = tmp_df['label'].tolist()
train_attacker_loader5 = mu.get_dataloader(tmp_df)

train_x = []
train_y = []
attacker_train_loader = []

train_x.append(shadow_train_x0)
train_x.append(shadow_train_x1)
train_x.append(shadow_train_x3)
train_x.append(shadow_train_x5)
train_y.append(shadow_train_y0)
train_y.append(shadow_train_y1)
train_y.append(shadow_train_y3)
train_y.append(shadow_train_y5)
attacker_train_loader.append(train_attacker_loader0)
attacker_train_loader.append(train_attacker_loader1)
attacker_train_loader.append(train_attacker_loader3)
attacker_train_loader.append(train_attacker_loader5)

# target model prediction
tmp_df = generator.reshape_for_attack(target_defense0)
target_train_x0 = tmp_df.data.tolist()
target_train_y0 = tmp_df['label'].tolist()
test_attacker_loader0 = mu.get_dataloader(tmp_df)

tmp_df = generator.reshape_for_attack(target_defense1)
target_train_x1 = tmp_df.data.tolist()
target_train_y1 = tmp_df['label'].tolist()
test_attacker_loader1 = mu.get_dataloader(tmp_df)

tmp_df = generator.reshape_for_attack(target_defense3)
target_train_x3 = tmp_df.data.tolist()
target_train_y3 = tmp_df['label'].tolist()
test_attacker_loader3 = mu.get_dataloader(tmp_df)

tmp_df = generator.reshape_for_attack(target_defense5)
target_train_x5 = tmp_df.data.tolist()
target_train_y5 = tmp_df['label'].tolist()
test_attacker_loader5 = mu.get_dataloader(tmp_df)

test_x = []
test_y = []
attacker_test_loader = []

test_x.append(target_train_x0)
test_x.append(target_train_x1)
test_x.append(target_train_x3)
test_x.append(target_train_x5)
test_y.append(target_train_y0)
test_y.append(target_train_y1)
test_y.append(target_train_y3)
test_y.append(target_train_y5)
attacker_test_loader.append(test_attacker_loader0)
attacker_test_loader.append(test_attacker_loader1)
attacker_test_loader.append(test_attacker_loader3)
attacker_test_loader.append(test_attacker_loader5)

# argment for mobilenetV3
args['classes'] = [0,1]
hyperargs = {
    'in_channels': 1,
    'kernel_size': (2,3),
    'stride': 1,
    'padding': 3
}
args['hyperargs'] = hyperargs

RF_attack_model = mm.RF()
DT_attack_model = mm.DT()
svm_attack = mm.SVM()

mobilenet_attack = mm.mobilenet(args=args, hyperargs = hyperargs)

defense_lst = [0,1,3,5]
for i in range(4):
    if defense_lst[i] != 0:
        print(f' - - defense {defense_lst[i]} - -')
    else:
        print(f' - - No Defense - - ')
    
    svm_attack.train_model(train_x[i], train_y[i])
    train_acc = svm_attack.model_acc(train_x[i], train_y[i])
    test_acc = svm_attack.model_acc(test_x[i], test_y[i])
    print(f'svm_attack acc (train)    : {train_acc}')
    print(f'svm_attack acc (test)     : {test_acc}')
    train_acc = svm_attack.test_model_auc(train_x[i], train_y[i])
    test_acc = svm_attack.test_model_auc(test_x[i], test_y[i])
    print(f'svm_attack auc (train)    : {train_acc}')
    print(f'svm_attack auc (test)     : {test_acc}')

    RF_attack_model.train_model(train_x[i], train_y[i])
    train_acc = RF_attack_model.model_acc(train_x[i], train_y[i])
    test_acc = RF_attack_model.model_acc(test_x[i], test_y[i])
    print(f'RF_attack acc (train)    : {train_acc}')
    print(f'RF_attack acc (test)     : {test_acc}')
    train_acc = RF_attack_model.test_model_auc(train_x[i], train_y[i])
    test_acc = RF_attack_model.test_model_auc(test_x[i], test_y[i])
    print(f'RF_attack auc (train)    : {train_acc}')
    print(f'RF_attack auc (test)     : {test_acc}')

    DT_attack_model.train_model(train_x[i], train_y[i])
    train_acc = DT_attack_model.model_acc(train_x[i], train_y[i])
    test_acc = DT_attack_model.model_acc(test_x[i], test_y[i])
    print(f'DT_attack acc (train)    : {train_acc}')
    print(f'DT_attack acc (test)     : {test_acc}')
    train_acc = DT_attack_model.test_model_auc(train_x[i], train_y[i])
    test_acc = DT_attack_model.test_model_auc(test_x[i], test_y[i])
    print(f'DT_attack auc (train)    : {train_acc}')
    print(f'DT_attack auc (test)     : {test_acc}')

    mobilenet_attack = mm.mobilenet(args=args, hyperargs = hyperargs)
    mobilenet_attack.train(attacker_train_loader[i], 
        mc.ROOT + mc.ATTACK+ mc.ATTACK_MODEL + mc.SET_NAME + f'MobileNetv3_defense_{defense_lst[i]}.pt')
    mobilenet_attack.load_model(
        mc.ROOT + mc.ATTACK+ mc.ATTACK_MODEL + mc.SET_NAME + f'MobileNetv3_defense_{defense_lst[i]}.pt')
    train_acc = mobilenet_attack.model_acc(attacker_train_loader[i])
    test_acc = mobilenet_attack.model_acc(attacker_test_loader[i])
    print(f'MobileNet accuracy (train)    : {train_acc}')
    print(f'MobileNet accuracy (test)     : {test_acc}')
    train_acc = mobilenet_attack.model_auc(attacker_train_loader[i])
    test_acc = mobilenet_attack.model_auc(attacker_test_loader[i])
    print(f'MobileNet auc (train)    : {train_acc}')
    print(f'MobileNet auc (test)     : {test_acc}')