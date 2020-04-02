import os 
import torch
from torchsummary import summary
from ptflops import get_model_complexity_info

from models import *

state_dict_path = "./checkpoints"
state_dict_list = os.listdir(state_dict_path)

def str_match(model_name):
    for name in state_dict_list:
        if model_name in name:
            return os.path.join(state_dict_path,name)

def main():
    model_list = ["fbnet_cb", "hardnet39ds", "hardnet68ds" , "mixnet_s", "mixnet_m", "mixnet_l", "mnasnet_a1", "mnasnet_b1", "shufflenetv2b_w3d2", "efficientnet_b0"]
    for idx in range(len(model_list)):
        model_name = model_list[idx]
        print("==> ",model_name)
        model = eval("%s()"%model_name)
        model.load_state_dict(torch.load(str_match(model_name)))
        summary(model,(3,224,224))

def analysis(model_name):
    model = eval("%s()"%model_name)
    model.load_state_dict(torch.load(str_match(model_name)))
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


if __name__=="__main__":
    main()
    analysis("fbnet_cb")