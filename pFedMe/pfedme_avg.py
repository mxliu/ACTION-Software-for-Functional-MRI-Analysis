import collections

def pFM_A(para_list,data_num_list,before_para,bata=2):
    # para_list: parameters of local models
    # data_num_list: sample number of local sites
    # before_para: parameters of the last round
    dict=collections.OrderedDict()
    rate=[x/sum(data_num_list) for x in data_num_list]

    for i in range(len(para_list)):
        for key in para_list[i].keys():
            para_list[i][key]=para_list[i][key]*rate[i]

    for i in range(len(para_list)):
        if i==0:
            dict=para_list[i]
        else:
            for n in dict.keys():
                dict[n]=dict[n]+para_list[i][n]

    for k in dict.keys():
        dict[k] = (1-bata)*before_para[k].cuda()+bata*dict[k].cuda()

    return dict

def FT_para(model, before_para, lr=1e-4):
    # model: current model
    # before_para: parameters of the last epoch
    para = model.state_dict()
    for k in para.keys():
        para[k] = para[k] - lr * (before_para[k] - para[k])
    model.load_state_dict(para)
    return model