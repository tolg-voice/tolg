import os
import sys
import timeit

from importlib.machinery import SourceFileLoader

import numpy as np
import torch

from dnnClasses import HiddenLayer


# run flags
make_dirs = 1
make_scp = 1
do_reaper_pitch_analysis = 0
do_sptk_pitch_analysis = 0
do_glott_vocoder_analysis = 1
make_dnn_train_data = 1
make_dnn_infofile = 1
do_dnn_training = 1
do_glott_vocoder_synthesis = 1

# directories
prjdir = './' # change to your own local dir
datadir = os.path.join(prjdir, 'dnn_demo', 'data')

# path to REAPER binary (optional)
reaper = 'reaper'
# paths to SPTK binaries (optional)
sox = 'sox'
pitch = 'pitch -a 0 -s 16.0 -p 80 '
x2x = 'x2x'

# GlottDNN binaries and default config
Analysis = prjdir + '/src/Analysis'
Synthesis = prjdir + '/src/Synthesis'
config_default = prjdir + '/dnn_demo/config_dnn_demo.cfg'

# general parameters
sampling_frequency = 16000
warping_lambda = 0.00
use_external_gci = False

# Neural net input params
inputs = ['f0', 'gain', 'hnr', 'slsf', 'lsf']
input_exts = ['.f0', '.gain', '.hnr', '.slsf','.lsf']
input_dims = [1, 1, 5, 10, 30] # set feature to zero if not used
outputs = ['pls']
output_exts = ['.pls']
output_dims = [400]

# dnn data conf
dnn_name = 'dnn_demo_slt'
train_data_dir = os.path.join(prjdir, 'nndata/traindata', dnn_name)
weights_data_dir = os.path.join(prjdir, 'nndata/weights', dnn_name)
remove_unvoiced_frames = True
validation_ratio = 0.10
test_ratio = 0.10
max_number_of_files = 1000

# dnn train conf
n_hidden = [150, 250, 300]
learning_rate = 1e-4
batch_size = 100
max_epochs = 100
patience = 5  # early stopping criterion
optimizer = 'adam'

# synthesis configs
use_dnn_generated_excitation = True


# Config file 
if len(sys.argv) < 2:
    sys.exit("Usage: python GlottDnnScript.py config.py")
if os.path.isfile(sys.argv[1]):
    conf = SourceFileLoader('', sys.argv[1]).load_module()
else:
    sys.exit("Config file " + sys.argv[1] + " does not exist")    

# Set torch device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')    


def load_data(filename, size2):
    var = np.fromfile(filename, dtype=np.float32, count=-1, sep='')
    var = np.reshape(var, (-1, size2))
    return torch.tensor(var).to(device)


def save_network(layerList, layer_out):
    fid = open( weights_data_dir + '/' + dnn_name + '.dnnData','w')   
    # hidden layers
    for layer in layerList[:-1]:
        layer.linear.weight.detach().numpy().astype(np.float32).T.tofile(fid)
        layer.linear.bias.detach().numpy().astype(np.float32).tofile(fid)
    # output layers
    layer_out = layerList[-1]
    layer_out.weight.detach().numpy().astype(np.float32).T.tofile(fid)
    layer_out.bias.detach().numpy().astype(np.float32).tofile(fid)


def list_dir_fullpath(dirname,start,extension):
    output = []
    for f in os.listdir(dirname):
        if f.endswith(extension) & f.startswith(start):
            output.append(os.path.join(dirname,f))
    return output


def evaluate_dnn(learning_rate=0.1, n_epochs=150,
                    n_in=42, n_out=500, n_hidden=[100, 250, 500], batch_size=32):

    num_hidden = len(n_hidden)
    dnn_name = 'dnn_demo_slt'
    prjdir = './' # change to your own local dir
    train_data_dir = os.path.join(prjdir, 'nndata/traindata', dnn_name)
    weights_data_dir = os.path.join(prjdir, 'nndata/weights', dnn_name)
    nndata_basename = train_data_dir + '/' + dnn_name
    
    # load data as torch.tensor
    valid_set_x = load_data(nndata_basename + '.val.idat', n_in)
    valid_set_y = load_data(nndata_basename + '.val.odat', n_out)
    train_set_x = load_data(nndata_basename + '.train.idat', n_in)
    train_set_y = load_data(nndata_basename + '.train.odat', n_out)

    # compute number of minibatches for training and validation
    n_train_batches = train_set_x.shape[0]
    n_valid_batches = valid_set_x.shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    print ('train batches %d' % (n_train_batches) ) 
    print ('valid batches %d' % (n_valid_batches) ) 

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print ('... building the model')
    print ('Number of hidden layers: %i' % (num_hidden))

    # pytorch
    layerList = []
    for i in range(0, num_hidden):
        if i > 0:
            cn_in = n_hidden[i-1]
            cn_out = n_hidden[i]
        else:
            cn_in = n_in
            cn_out = n_hidden[0]
        print ('in: %d, out: %d' % (cn_in, cn_out))    
        cLayer = HiddenLayer(n_in=cn_in, n_out=cn_out)
        layerList.append(cLayer)

    # output layer
    layer_out = torch.nn.Linear(n_hidden[-1], n_out, bias=True)
    layerList.append(layer_out)       

    # model 
    model = torch.nn.Sequential(*layerList)
    model = model.to(device)

    # loss
    criterion = torch.nn.MSELoss()
  
    params = model.parameters()
    # Original papers used SGD optimizer
    #optim = torch.optim.SGD(params, lr=learning_rate)
    # Adam is notably faster
    optim = torch.optim.Adam(params, lr=learning_rate)

    def train_model(index):
        x = train_set_x[index * batch_size: (index + 1) * batch_size]
        y = train_set_y[index * batch_size: (index + 1) * batch_size]
        optim.zero_grad()
        y_hat = model(x)
        loss = criterion(y, y_hat)
        loss.backward()
        optim.step()
        return loss

    def validate_model(index):
        x = valid_set_x[index * batch_size: (index + 1) * batch_size]
        y = valid_set_y[index * batch_size: (index + 1) * batch_size]
        y_hat = model(x)
        loss = criterion(y, y_hat)
        return loss


    ###############
    # TRAIN MODEL #
    ###############
    print ('... training')
    # early-stopping parameters
    patience = 5  # early stopping criterion
    patience = patience

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        permutation = np.random.permutation(n_train_batches)
        training_losses = []            
        # loop over minibatches
        for minibatch_index in range(n_train_batches):
            
            iter_ind = (epoch - 1) * n_train_batches + minibatch_index  
            index = permutation[minibatch_index]
            loss = train_model(index)
            training_losses.append(loss.detach().numpy())

            if iter_ind % 100 == 0:
                print('     epoch %d, minibatch %d / %d' % (epoch, minibatch_index + 1, n_train_batches))

        this_training_loss = np.mean(training_losses)

        # compute loss on validation set
        validation_losses = [validate_model(i).detach().numpy() for i in range(n_valid_batches)]
        this_validation_loss = np.mean(validation_losses)
		
        print('epoch %i, minibatch %i/%i, training error %f, validation error %f' % (epoch, minibatch_index + 1, n_train_batches, this_training_loss, this_validation_loss)) 
                
        # if we got the best validation score until now
        if this_validation_loss < best_validation_loss:
            # reset patience
            patience = patience 
            # save best validation score and iteration number
            best_validation_loss = this_validation_loss
            # test it on the test set      
            save_network(layerList, layer_out)
        else:
            patience -= 1    
            
        if patience == 0:
            done_looping = True
    	

    end_time = timeit.default_timer()
    print('Optimization complete.')
 
    print(('The code ran for %.2fm' % ((end_time - start_time) / 60.)))


