
##########################################################
# HCGS sweep
# Deepak Kadetotad
# Arizona State University
# November, 2018
##########################################################


import os
import sys
import glob
import configparser
import numpy as np
from utils import check_cfg,create_chunks,write_cfg_chunk,compute_avg_performance, \
                  read_args_command_line, run_shell,compute_n_chunks, get_all_archs,cfg_item2sec, \
                  dump_epoch_results, run_shell_display, create_curves
import re
from distutils.util import strtobool

def write_out_fol(config, i, drop, block_size, out_folder, k):
    # out_folder = 'exp/TIMIT_LSTM_fmllr_' + str(i) + 'c_uni_2l_hcgs_' + str(drop[0]) + '_' + str(
    #     block_size[0]) + '_' + str(drop[1]) + '_' + str(block_size[1])
    config['exp']['out_folder'] = out_folder
    config['architecture1']['hcgsx_block'] = str(block_size[0])+','+str(block_size[1])
    config['architecture1']['hcgsh_block'] = str(block_size[0])+','+str(block_size[1])
    config['architecture1']['hcgsx_drop'] = str(drop[0])+','+str(drop[1])
    config['architecture1']['hcgsh_drop'] = str(drop[0])+','+str(drop[1])
    config['architecture1']['lstm_lay'] = str(i)+','+str(i)
    config['architecture1']['param_quant'] = str(k)+','+str(k)
    config['architecture2']['param_quant'] = str(k)
    config_file = 'cfg/TIMIT/run_sweep.cfg'
    with open(config_file, 'w') as configfile:
        config.write(configfile)
    cmd_sweep='python run_exp.py ' + config_file
    run_shell_display(cmd_sweep)
    return 1

# Reading chunk-specific cfg file (first argument-mandatory file)
cfg_file = sys.argv[1]

if not (os.path.exists(cfg_file)):
    sys.stderr.write('ERROR: The config file %s does not exist!\n' % (cfg_file))
    sys.exit(0)
else:
    config = configparser.ConfigParser()
    config.read(cfg_file)

cell_size = [256]
total_drop = [75, 87.5, 93.75]
drop = [50, 50]
block_size = [64, 4]
quant = [3]

for i in cell_size:
    for k in quant:
        for j in total_drop:
            if j == 50:
                drop = [50, 0]
                block_size[0] = i / 8
                while block_size[0] != 2:
                    out_folder = 'exp/TIMIT_LSTM_fmllr_' + str(i) + 'c_uni_2l_hcgs_' + str(drop[0]) + 'd' + str(
                        block_size[0]) + 'b' + '_quant' + str(k) + 'b'
                    x = write_out_fol(config, i, drop, block_size, out_folder, k)
                    block_size[0] = block_size[0] / 2
                block_size = [64, 4]

            elif j == 75:
                # drop = [75, 0]
                # block_size[0] = i / 8
                # while block_size[0] != 2:
                #     out_folder = 'exp/TIMIT_LSTM_fmllr_' + str(i) + 'c_uni_2l_hcgs_' + str(drop[0]) + 'd' + str(
                #         block_size[0]) + 'b' + '_quant' + str(k) + 'b'
                #     x = write_out_fol(config, i, drop, block_size, out_folder, k)
                #     block_size[0] = block_size[0] / 2
                # block_size = [64, 4]
                drop = [50, 50]
                block_size[0] = i / 8
                while block_size[0] != 16:
                    block_size[1] = block_size[0] / 2
                    while block_size[1] != 2:
                        out_folder = 'exp/TIMIT_LSTM_fmllr_' + str(i) + 'c_uni_2l_hcgs_' + str(drop[0]) + 'd' + str(
                            block_size[0]) + 'b_' + str(drop[1]) + 'd' + str(block_size[1]) + 'b' + '_quant' + str(k) + 'b'
                        x = write_out_fol(config, i, drop, block_size, out_folder, k)
                        block_size[1] = block_size[1] / 2
                    block_size[0] = block_size[0] / 2
                block_size = [64, 4]

            elif j == 87.5:
                # drop = [87.5, 0]
                # block_size[0] = i / 8
                # while block_size[0] != 2:
                #     out_folder = 'exp/TIMIT_LSTM_fmllr_' + str(i) + 'c_uni_2l_hcgs_' + str(drop[0]) + 'd' + str(
                #         block_size[0]) + 'b' + '_quant' + str(k) + 'b'
                #     x = write_out_fol(config, i, drop, block_size, out_folder, k)
                #     block_size[0] = block_size[0] / 2
                # block_size = [64, 4]
                drop = [50, 75]
                block_size[0] = i / 8
                while block_size[0] != 16:
                    block_size[1] = block_size[0] / 4
                    while block_size[1] != 2:
                        out_folder = 'exp/TIMIT_LSTM_fmllr_' + str(i) + 'c_uni_2l_hcgs_' + str(drop[0]) + 'd' + str(
                            block_size[0]) + 'b_' + str(drop[1]) + 'd' + str(block_size[1]) + 'b' + '_quant' + str(k) + 'b'
                        x = write_out_fol(config, i, drop, block_size, out_folder, k)
                        block_size[1] = block_size[1] / 2
                    block_size[0] = block_size[0] / 2
                block_size = [64, 4]
                # drop = [75, 50]
                # block_size[0] = i / 8
                # while block_size[0] != 16:
                #     block_size[1] = block_size[0] / 2
                #     while block_size[1] != 2:
                #         out_folder = 'exp/TIMIT_LSTM_fmllr_' + str(i) + 'c_uni_2l_hcgs_' + str(drop[0]) + 'd' + str(
                #             block_size[0]) + 'b_' + str(drop[1]) + 'd' + str(block_size[1]) + 'b' + '_quant' + str(k) + 'b'
                #         x = write_out_fol(config, i, drop, block_size, out_folder, k)
                #         block_size[1] = block_size[1] / 2
                #     block_size[0] = block_size[0] / 2
                # block_size = [64, 4]

            elif j == 93.75:
                # drop = [93.75, 0]
                # block_size[0] = i / 16
                # while block_size[0] != 2:
                #     out_folder = 'exp/TIMIT_LSTM_fmllr_' + str(i) + 'c_uni_2l_hcgs_' + str(drop[0]) + 'd' + str(
                #         block_size[0]) + 'b' + '_quant' + str(k) + 'b'
                #     x = write_out_fol(config, i, drop, block_size, out_folder, k)
                #     block_size[0] = block_size[0] / 2
                # block_size = [64, 4]
                drop = [75, 75]
                block_size[0] = i / 8
                while block_size[0] != 16:
                    block_size[1] = block_size[0] / 4
                    while block_size[1] != 2:
                        out_folder = 'exp/TIMIT_LSTM_fmllr_' + str(i) + 'c_uni_2l_hcgs_' + str(drop[0]) + 'd' + str(
                            block_size[0]) + 'b_' + str(drop[1]) + 'd' + str(block_size[1]) + 'b' + '_quant' + str(k) + 'b'
                        x = write_out_fol(config, i, drop, block_size, out_folder, k)
                        block_size[1] = block_size[1] / 2
                    block_size[0] = block_size[0] / 2
                block_size = [64, 4]
                # drop = [50, 87.5]
                # block_size[0] = i / 8
                # while block_size[0] != 16:
                #     block_size[1] = block_size[0] / 8
                #     while block_size[1] != 2:
                #         out_folder = 'exp/TIMIT_LSTM_fmllr_' + str(i) + 'c_uni_2l_hcgs_' + str(drop[0]) + 'd' + str(
                #             block_size[0]) + 'b_' + str(drop[1]) + 'd' + str(block_size[1]) + 'b'
                #         x = write_out_fol(config, i, drop, block_size, out_folder, k)
                #         block_size[1] = block_size[1] / 2
                #     block_size[0] = block_size[0] / 2
                # block_size = [64, 4]
                # drop = [87.5, 50]
                # block_size[0] = i / 8
                # while block_size[0] != 16:
                #     block_size[1] = block_size[0] / 2
                #     while block_size[1] != 2:
                #         out_folder = 'exp/TIMIT_LSTM_fmllr_' + str(i) + 'c_uni_2l_hcgs_' + str(drop[0]) + 'd' + str(
                #             block_size[0]) + 'b_' + str(drop[1]) + 'd' + str(block_size[1]) + 'b'
                #         x = write_out_fol(config, i, drop, block_size, out_folder, k)
                #         block_size[1] = block_size[1] / 2
                #     block_size[0] = block_size[0] / 2
                # block_size = [64, 4]
