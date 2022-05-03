import torch
from utils import get_student, get_feat_info
from local_structure import get_local_model
import copy


def collect_model(args, data_info):
    device = torch.device("cpu") if args.gpu<0 else torch.device("cuda:" + str(args.gpu))
    
    feat_info = get_feat_info(args)
    
    s_model = []
    s_model_optimizer = []
    model1 = get_student(args, data_info)
    s_model.append(model1)
    s_model_optimizer.append(torch.optim.Adam(s_model[0].parameters(), lr=args.lr, weight_decay=args.weight_decay))
    s_model[0].to(device)
    for i in range(args.model_num-1):
        model = copy.deepcopy(model1)
        s_model.append(model)
        s_model_optimizer.append(torch.optim.Adam(s_model[i+1].parameters(), lr=args.lr, weight_decay=args.weight_decay))
        s_model[i+1].to(device)

    local_model = get_local_model(feat_info);                       local_model.to(device)

    # construct optimizers
    local_model_optimizer = None #torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    
    # construct model dict
    model_dict = {}
    model_dict['s_model'] = {'model':s_model, 'optimizer':s_model_optimizer}
    model_dict['local_model'] = {'model':local_model, 'optimizer':local_model_optimizer}
    return model_dict