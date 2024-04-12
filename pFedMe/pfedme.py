# T Dinh, C., Tran, N., Nguyen, J., 2020. Personalized federated learning with moreau envelopes. Advances in Neural Information Processing Systems 33, 21394–21405.

import random
import os
import scipy.io as io
from tqdm import tqdm
from pFedMe.options import parse
from pFedMe.dataset import Load_Data
from pFedMe.model import Module_1, Classifier
from pFedMe.pfedme_avg import FT_para, pFM_A
import torch
import numpy as np

# init
argv = parse()

def acc(Label, Pred):
    true_list = np.array([Label[i] - Pred[i] for i in range(len(Label))])
    true_list[true_list != 0] = 1
    err = true_list.sum()
    acc = (len(Label)-err)/len(Label)
    return acc


# Set seed and device
torch.manual_seed(argv.seed)
np.random.seed(argv.seed)
random.seed(argv.seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(argv.seed)
else:
    device = torch.device("cpu")


# dataset
# Here, we take three sites as an example, but in practice, any number of sites is allowed.
# Please refer to the demo data and place your data files and label files for each site in their respective folders.
data_dir = argv.data_dir
label_dir = argv.label_dir
data_file = os.listdir(data_dir)
label_file = os.listdir(label_dir)
site_num = len(data_file)
sample_num_list = []
for num in range(site_num):
    data_path = data_dir + '/' + data_file[num]
    label_path = label_dir + '/' + label_file[num]
    exec ("dataset%s=Load_Data(data_path, label_path, k_fold=argv.k_fold)" % num)
    exec ("dataset_test%s=Load_Data(data_path, label_path, k_fold=argv.k_fold)" % num)
    exec ("sample_num_list.append(len(dataset%s))" % num)
    exec ("dataloader%s=torch.utils.data.DataLoader(dataset%d, batch_size=argv.minibatch_size, shuffle=False)" % (num,num))
    exec ("dataloader_test%s=torch.utils.data.DataLoader(dataset_test%d, batch_size=1, shuffle=False)" % (num, num))


# train
def step(model, c_model, criterion, data, label, device='cpu', optimizer=None, c_optimizer=None):
    # model: encoder, c_model: classifier
    # criterion: loss
    # data: samples, label: labels
    if optimizer is None:
        model.eval(), c_model.eval()
    else:
        model.train(), c_model.train()

    # run model
    latent = model(data.to(device))
    logit = c_model(latent)
    loss = criterion(logit, label.to(device))

    # optimize model
    if optimizer is not None:
        optimizer.zero_grad()
        if c_optimizer is not None:
            c_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if c_optimizer is not None:
            c_optimizer.step()

    return logit, loss


# server
# You can replace the model as needed.
# You are free to decide here whether to use the pre-trained parameters we provide.
model_init = Module_1(input_dim=116, hidden_dim=64)
parameter_init = model_init.state_dict()
parameter_init_fix = model_init.state_dict()
c_model_init = Classifier(latent_dim=64, num_classes=2)
c_parameter_init = c_model_init.state_dict()
c_parameter_init_fix = c_model_init.state_dict()


# fl train
R = 0.0
for k in range(argv.k_fold):
    R_k = []
    lr = argv.lr
    for i in range(site_num):
        exec('L%s=[]' % i)

    # train
    for iter_num in range(argv.num_iters):
        para_list = []
        c_para_list=[]
        for site in range(site_num):
            exec('dataset%s.set_fold(k,train=True)'%site)
            exec ("dataloader=dataloader%s"%site)
            model = Module_1(input_dim=116, hidden_dim=64)
            model.to(device)
            model.load_state_dict(parameter_init)  # read server-side parameters
            c_model = Classifier(latent_dim=64, num_classes=2)
            c_model.to(device)
            c_model.load_state_dict(c_parameter_init)  # read server-side parameters

            criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            optimizer_c = torch.optim.Adam(c_model.parameters(), lr=lr)

            for epoch in range(argv.num_epochs):
                loss_accumulate = 0.0
                # Save the model parameters of the previous epoch
                e_model_para = model.state_dict()
                e_c_model_para = c_model.state_dict()
                Label = []
                Pred = []
                for i, x in enumerate(tqdm(dataloader, ncols=80, desc=f'k:{k},i:{iter_num},s:{site},e:{epoch}')):
                    data = x['X'].permute(0, 2, 1)
                    label = x['y']
                    Label.extend(label.tolist())

                    logit, loss = step(model=model,
                                       c_model=c_model,
                                       criterion=criterion,
                                       data=data,
                                       label=label,
                                       device=device,
                                       optimizer=optimizer,
                                       c_optimizer=optimizer_c)

                    pred = logit.argmax(1)
                    prob = logit.softmax(1)
                    Pred.extend(pred.tolist())
                    loss_accumulate += loss.detach().cpu().numpy()  # cross-entropy loss

                exec('L%s.append(loss_accumulate)' % site)
                Acc = acc(Label,Pred)
                # fine tuning
                model = FT_para(model, e_model_para)
                p_model = FT_para(c_model, e_c_model_para)
                print('acc:',Acc)

            # save model
            if iter_num == argv.num_iters - 1:
                torch.save(model.state_dict(),
                           argv.save_dir + '/' + 'site' + str(site) + '_fold' + str(k) + '_model.pth')
                torch.save(c_model.state_dict(),
                           argv.save_dir + '/' + 'site' + str(site) + '_fold' + str(k) + '_c_model.pth')

            para_list.append(model.state_dict())
            c_para_list.append(c_model.state_dict())


        # federated aggregation with moreau envelopes in pfedme
        parameter_init = pFM_A(para_list, sample_num_list, parameter_init)
        pFM_A(c_para_list, sample_num_list, c_parameter_init)

        # test
        for site in range(site_num):
            exec('dataset_test%s.set_fold(k,train=False)' % site)
            exec("dataloader_test=dataloader_test%s"%site)

            model = Module_1(input_dim=116, hidden_dim=64)
            model.to(device)
            model.load_state_dict(parameter_init)
            c_model = Classifier(latent_dim=64, num_classes=2)
            c_model.to(device)
            c_model.load_state_dict(c_parameter_init)
            model.eval()
            c_model.eval()
            criterion = torch.nn.CrossEntropyLoss()
            Label = []
            Pred = []

            if iter_num == argv.num_iters - 1: Prob = []
            for i, x in enumerate(tqdm(dataloader_test, ncols=80, desc=f'k:{k},i:{iter_num},s:{site}')):
                data = x['X'].permute(0, 2, 1)
                label = x['y']
                Label.append(label.item())

                logit, loss = step(model=model,
                                   c_model=c_model,
                                   criterion=criterion,
                                   data=data,
                                   label=label,
                                   device=device)

                pred = logit.argmax(1)
                prob = logit.softmax(1)
                Pred.append(pred.item())
                if iter_num == argv.num_iters - 1: Prob.append(prob.tolist())
            Acc = acc(Label, Pred)
            print('k:', k, 'i:', iter_num, 's:', site, 'result:')
            print(Acc)
            R_k.append(Acc)

            # save results
            if iter_num == argv.num_iters - 1:
                Label = np.array(Label)
                Pred = np.array(Pred)
                Prob = torch.tensor(Prob).squeeze(1).numpy()
                np.save(argv.save_dir + '/' + 'site' + str(site) + '_fold' + str(k) + '_Label', Label)
                np.save(argv.save_dir + '/' + 'site' + str(site) + '_fold' + str(k) + '_Pred', Pred)
                np.save(argv.save_dir + '/' + 'site' + str(site) + '_fold' + str(k) + '_Prob', Prob)

    # save fold results
    R_k = np.array(R_k).reshape(argv.num_iters, site_num)
    io.savemat(argv.save_dir + '/' + 'fold' + str(k) + '_result.mat', {'acc': R_k})

    # save loss
    for i in range(site_num):
        exec('L%s=np.array(L%d)' % (i, i))
        exec("np.save(argv.save_dir + '/' + 'site' + str(i) + '_fold' + str(k) + '_Loss', L%s)" % i)
    print('fold ', str(k), 'result:')
    print(R_k)
    R += R_k
    parameter_init = parameter_init_fix
    c_parameter_init = c_parameter_init_fix

print('final result:')
print(R/argv.k_fold)

# save final result
io.savemat(argv.save_dir+'/'+'final_result.mat', {'acc': R/argv.k_fold})
