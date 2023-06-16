import sys
import os
import numpy as np
import random
from importlib.machinery import SourceFileLoader
import os

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


# GlottDNN binaries and default config
Analysis = prjdir + '/src/Analysis'
Synthesis = prjdir + '/src/Synthesis'
config_default = prjdir + '/dnn_demo/config_dnn_demo.cfg'

# general parameters
sampling_frequency = 16000
warping_lambda = 0.00
use_external_gci = False

# Neural net input params
# inputs = ['f0', 'gain', 'hnr', 'slsf', 'lsf', 'rd']
# input_exts = ['.f0', '.gain', '.hnr', '.slsf','.lsf', '.rd']
# input_dims = [1, 1, 5, 10, 30, 1] # set feature to zero if not used


inputs = ['f0']
input_exts = ['.f0']
input_dims = [1, 1] # set feature to zero if not used
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



if do_dnn_training:
    import TrainDnn


def mkdir_p(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def make_directories():
    # Prepare environment
    dirs = ['wav',
            'raw',
            'gci',
            'rd',
            'scp',
            'exc',
            'syn',
            'f0',
            'gain',
            'lsf',
            'slsf',
            'hnr',
            'pls']

    for d in dirs:
        mkdir_p(os.path.join(datadir, d))

    for t in inputs:
        mkdir_p(datadir + '/' + t)
    for t in outputs:
        mkdir_p(datadir + '/' + t)

    # Dnn directiories
    mkdir_p(train_data_dir)
    mkdir_p(weights_data_dir)

def make_file_lists():

    scp_types = ['wav']
    scp_types.extend(inputs)
    scp_types.extend(outputs)

    extensions = ['.wav']
    extensions.extend(input_exts)
    extensions.extend(output_exts)

    for t,e in zip(scp_types, extensions):

        scpfile = open(datadir + '/scp/' + t + '.scp','w')
        for f in sorted(set(os.listdir(datadir + '/' + t))):
            if f.endswith(e):
                scpfile.write(os.path.abspath(datadir + '/' + t + '/' + f + '\n'))
        scpfile.close()


def glott_vocoder_analysis():
    wavscp = datadir + '/scp/wav.scp'
    with open(wavscp,'r') as wavfiles:
        for file in wavfiles:
            wavfile = file.rstrip()
            if os.path.isfile(wavfile):
                # define file paths
                bname = os.path.splitext(os.path.basename(wavfile))[0]
                f0file = os.path.join(datadir, 'f0', bname + '.f0')
                gcifile = datadir + '/gci/' + bname + '.GCI'
                # create temporary analysis config for file

                # run analysis program
                cmd = f"export LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']};"
                cmd += Analysis + ' ' + wavfile + ' ' + config_default + ' ' + config_user
                os.system(cmd)
                # remove temporary config
                os.remove(config_user)



def package_data():
    # read and shuffle wav filelist
    wavscp = datadir + '/scp/wav.scp'
    with open(wavscp,'r') as wavfiles:
        filelist = wavfiles.read().splitlines()
    random.shuffle(filelist)

    if max_number_of_files < len(filelist):
        filelist = filelist[0:max_number_of_files]

    # initialize global min and max
    in_min = 9999*np.ones([1,sum(input_dims)],dtype=np.float32)
    in_max = -9999*np.ones([1,sum(input_dims)],dtype=np.float32)

    n_frames = np.zeros([len(filelist)], dtype='int')
    for file_idx, wavfile in enumerate(filelist):
        if os.path.isfile(wavfile):
            bname = os.path.splitext(os.path.basename(wavfile))[0]
            # print (bname)
            f0_file = datadir + '/f0/' + bname + '.f0'
            n_frames[file_idx] = (np.fromfile(f0_file, dtype=np.float32, count=-1, sep='')).shape[0]
            # allocate file data
            input_data = np.empty([n_frames[file_idx], sum(input_dims)], dtype=np.float32)
            feat_start = 0
            for (ftype, ext, dim) in zip( inputs, input_exts, input_dims):
                if dim > 0:
                    # read feat  data
                    # print(ftype)
                    feat_file = datadir + '/'+ ftype + '/' + bname + ext
                    feat = np.fromfile(feat_file, dtype=np.float32, count=-1, sep='')
                    # check length is multiple of feature dimension
                    # print(feat)
                    # print(dim)
                    # assert len(feat) % dim == 0, \
                    #     " Length mismatch for " + ftype
                    # # reshape
                    feat = np.reshape(feat, (-1,dim))
                    # set to input data matrix
                    print("feat", feat)
                    print("feat.shape",feat.shape)
                    # print(input_data[:,feat_start:feat_start+dim ])
                    print("feat_start", feat_start)
                    print("dim", dim)
                    print("feat_start+dim", feat_start+dim)
                # try:
                #     input_data[:,feat_start:feat_start+dim ] = feat
                # except ValueError as e:
                #     print("Error reading " + feat_file)
                #     print("Check that the feature sizes match between vocoder and python configs")
                #     raise e
                #     feat_start += dim
            # # remove unvoiced frames if requested
            # if remove_unvoiced_frames:
            #     input_data = input_data[input_data[:,0] > 0,:]
            # update global min and max
            in_min = np.minimum(np.amin(input_data, axis=0), in_min)
            in_max = np.maximum(np.amax(input_data, axis=0), in_max)

    new_min = 0.1
    new_max = 0.9

    n_val = round(validation_ratio * len(filelist))
    n_test = round(test_ratio * len(filelist))
    n_train = len(filelist) - n_val - n_test
    if n_train < 0:
        print ("oops")
    set_name = ['train', 'val', 'test']
    set_sizes = [n_train , n_val, n_test]
    print (set_sizes)

    set_file_counter = 1
    set_index = 0
    in_fid = open(train_data_dir + '/' + dnn_name + '.' + set_name[set_index] + '.idat' ,'w')
    out_fid = open(train_data_dir + '/' + dnn_name + '.' + set_name[set_index] + '.odat' ,'w')

    for file_idx, wavfile in enumerate(filelist):

        if set_file_counter > set_sizes[set_index]:
            set_file_counter = 1
            set_index += 1
            in_fid.close()
            out_fid.close()
            if set_sizes[set_index] == 0:
                set_index += 1
                continue
            else:
                in_fid = open(train_data_dir + '/' + dnn_name + '.' + set_name[set_index] + '.idat' ,'w')
                out_fid = open(train_data_dir + '/' + dnn_name + '.' + set_name[set_index] + '.odat' ,'w')

        if os.path.isfile(wavfile):
            bname = os.path.splitext(os.path.basename(wavfile))[0]

            # allocate input and output data
            input_data = np.empty([n_frames[file_idx], sum(input_dims)], dtype=np.float32)
            output_data = np.empty([n_frames[file_idx], sum(output_dims)], dtype=np.float32)
            # read input data
            feat_start = 0
            for (ftype, ext, dim) in zip(inputs, input_exts, input_dims):
                if dim > 0:
                    feat_file = datadir + '/' + ftype + '/' + bname + ext
                    feat = np.fromfile(feat_file, dtype=np.float32, count=-1, sep='')
                    feat = np.reshape(feat, (-1, dim))

                    if feat.shape[0] > input_data.shape[0]:
                        print("Error reading " + feat_file)
                        print("Check that the feature sizes match between vocoder and python configs")
                        continue
                    else:
                        input_data[:feat.shape[0], feat_start:feat_start+dim] = feat

                    feat_start += dim

            feat_start = 0
            for (ftype, ext, dim) in zip(outputs, output_exts, output_dims):
                if dim > 0:
                    feat_file = datadir + '/' + ftype + '/' + bname + ext
                    feat = np.fromfile(feat_file, dtype=np.float32, count=-1, sep='')
                    feat = np.reshape(feat, (-1, dim))

                    if feat.shape[0] > output_data.shape[0]:
                        print("Error reading " + feat_file)
                        print("Check that the feature sizes match between vocoder and python configs")
                        continue
                    else:
                        output_data[:feat.shape[0], feat_start:feat_start+dim] = feat

                    feat_start += dim

            # # remove unvoiced frames if requested
            # if remove_unvoiced_frames:
            #     output_data = output_data[input_data[:,0] > 0,:]
            #     input_data = input_data[input_data[:,0] > 0,:]

            # normalize and write input data
            input_data = (input_data - in_min) / (in_max - in_min) * (new_max - new_min) + new_min
            input_data.astype(np.float32).tofile(in_fid, sep='',format="%f")

            # write output data
            output_data.astype(np.float32).tofile(out_fid, sep='',format="%f")

            set_file_counter += 1

    # close files
    in_fid.close()
    out_fid.close()

    # write input min and max
    fid = open(weights_data_dir + '/' + dnn_name + '.dnnMinMax' ,'w')
    in_min.astype(np.float32).tofile(fid, sep='', format="%f")
    in_max.astype(np.float32).tofile(fid, sep='', format="%f")
    fid.close()

def main(argv):

    # Make directories
    if make_dirs:
        make_directories()

    # Make SCP list for wav
    if make_scp:
        make_file_lists()

    #
    # # GlottDNN Analysis
    # if do_glott_vocoder_analysis:
    #     glott_vocoder_analysis()

    # Package data for DNN training
    if make_dnn_train_data:
        package_data()

    # # Write Dnn infofile
    # if make_dnn_infofile:
    #     write_dnn_infofile()

    # Train Dnn with torch
    if do_dnn_training:
        dim_in = sum(input_dims)
        dim_out = sum(output_dims)
        TrainDnn.evaluate_dnn(n_in=dim_in, n_out=dim_out, n_hidden=n_hidden, batch_size=batch_size,
                 learning_rate=learning_rate, n_epochs = max_epochs)

    # Copy-synthesis
    # if do_glott_vocoder_synthesis:
    #     glott_vocoder_synthesis()
    
if __name__ == "__main__":
    main(sys.argv[1:])
