import collections

# weighted average
def W_A(para_list,data_num_list):
    # para_list: parameters of local models
    # data_num_list: sample number of local sites
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

    return dict