from argment import argparser
from os import path
import os

args = argparser()



SHADOW_MODEL = "shadow_model/"
ATTACK_MODEL = "attack_model/"
ORIGIN_MODEL = "origin_model/"
UNLEARN_MODEL = "unlearn_model/"
DEFENSE_MODEL = "defense_model/"

ROOT = "./temp_data/"
ATTACK = "for_attacker/"
TARGET = "for_target/"
DEFENSE = "for_defender/"

DATASET_PATH = "./DataSet/"
SET_NAME = args["set_name"] + '/'
NET_NAME = args["net_name"] + '/'

def path_create():
    # create root folder
    if not path.exists(ROOT):
        os.mkdir(ROOT)

    # create attacker folder
    if not path.exists(ROOT + ATTACK):
        os.mkdir(ROOT + ATTACK)

    if not path.exists(ROOT + ATTACK + ATTACK_MODEL):
        os.mkdir(ROOT + ATTACK + ATTACK_MODEL)

    ATTACK_SHADOW = ROOT + ATTACK + SHADOW_MODEL
    if not path.exists(ATTACK_SHADOW):
        os.mkdir(ATTACK_SHADOW)

    if not path.exists(ATTACK_SHADOW + SET_NAME):
        os.mkdir(ATTACK_SHADOW + SET_NAME)

    if not path.exists(ATTACK_SHADOW + SET_NAME + ORIGIN_MODEL):
        os.mkdir(ATTACK_SHADOW + SET_NAME + ORIGIN_MODEL)

    if not path.exists(ATTACK_SHADOW + SET_NAME + UNLEARN_MODEL):
        os.mkdir(ATTACK_SHADOW + SET_NAME + UNLEARN_MODEL)
    
    if not path.exists(ROOT + ATTACK + SHADOW_MODEL + DEFENSE_MODEL):
        os.mkdir(ROOT + ATTACK + SHADOW_MODEL + DEFENSE_MODEL)

    # create target folder
    if not path.exists(ROOT + TARGET):
        os.mkdir(ROOT + TARGET)

    if not path.exists(ROOT + TARGET + SET_NAME):
        os.mkdir(ROOT + TARGET + SET_NAME)

    if not path.exists(ROOT + TARGET + SET_NAME + ORIGIN_MODEL):
        os.mkdir(ROOT + TARGET + SET_NAME + ORIGIN_MODEL)

    if not path.exists(ROOT + TARGET + SET_NAME + UNLEARN_MODEL):
        os.mkdir(ROOT + TARGET + SET_NAME + UNLEARN_MODEL)

    if not path.exists(ROOT + TARGET + SET_NAME + DEFENSE_MODEL):
        os.mkdir(ROOT + TARGET + SET_NAME + DEFENSE_MODEL)

    # create dataset folder
    if not path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)
    
    if not path.exists(DATASET_PATH + SET_NAME):
        os.mkdir(DATASET_PATH + SET_NAME)
