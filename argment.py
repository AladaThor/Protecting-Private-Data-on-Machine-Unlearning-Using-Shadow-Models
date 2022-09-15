import argparse

def argparser():
    parser = argparse.ArgumentParser(description='program argment')
    parser.add_argument('--origin_sample_num', default='1000', type=int)
    parser.add_argument('--origin_model_num', default='20', type=int)
    parser.add_argument('--unlearn_model_num', default='10', type=int)
    parser.add_argument('--unlearn_sample_num', default='20', type=int)
    parser.add_argument('--net_name', default='resnet50', type=str)
    parser.add_argument('--set_name', default='cifar10', type=str)
    parser.add_argument('--epochs', default='45', type=int)
    parser.add_argument('--batch_size',default='32', type=int)
    parser.add_argument('--set_split_percentage', default='0.5', type=float)
    parser.add_argument('--class_num', nargs="?", const="default", type=int)
    parser.add_argument('--classes', nargs="?", const="default", type=list)
    parser.add_argument('--hyperargs', nargs="?", const="default", help="Hyper argments for MobileNet V3")
    parser.add_argument('-w', '--weight', default='0.7', type=float)
    parser.add_argument('--mode', default='0', type=int)
    parser.add_argument('--mode_name', default='OU', type=str)
    parser.add_argument('-dfp','--data_file_path',  nargs="?", const="default", type=str)
    
    x = parser.parse_args()

    return vars(x)

