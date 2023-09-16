import subprocess
import random
import string
import os
from configs import config
import argparse
from utils.debugger import MyDebugger
from copy import deepcopy

def get_in_list(args_list, arg_to_search):
    for arg in args_list:
        if arg[0] == arg_to_search:
            return arg[1]
    return None

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def get_prev_iter_informations(previous_iter_folder, inputs_dict):
    logfile = open(os.path.join(config.debug_base_folder, previous_iter_folder, 'print.log'))
    lines = [line.strip('\n\r') for line in logfile.readlines()]
    for line in lines:
        for key in inputs_dict.keys():
            if MyDebugger.get_save_text(key) in line:
                splitted_text = line.split(" ")
                current_epoch = int(splitted_text[1]) + 1  ## get epoch
                saved_path = splitted_text[-1]
                inputs_dict[key] = get_in_list(optional_args, key)(saved_path)
                inputs_dict['starting_epoch'] = current_epoch
                print(f"New epoch detected :{inputs_dict['starting_epoch']}")

    return inputs_dict


def get_previous_iteration_folder(random_string):
    previous_iter_folder = None
    # print(config.debug_base_folder)
    folder_list = subprocess.check_output(["ls", f"{config.debug_base_folder}"]).decode('utf-8')
    for folder in folder_list.split('\n'):
        if random_string in folder:
            previous_iter_folder = folder
    # print(folder_list.split('\n'))
    assert previous_iter_folder is not None
    return previous_iter_folder


optional_args = [("network_resume_path", str), ("optimizer_resume_path", str), ("save_Dmodel_path", str), ("starting_epoch", int), ("starting_stage", int),
                 ("resume_path", str), ("discriminator_resume_path", str), ("discriminator_opt_resume_path", str), ("scaler_resume_path", str)]
other_args = [("exp_id", int)]

if __name__ == '__main__':
    length = 20
    random_string = get_random_string(length)
    running_cnt = 0

    ## additional args for parsing
    job_args = [("use_fu_gpu", bool, False), ("gpu_count", int, 4), ("cpu_ratio", int, 4), ("partition", str, ''), ("run_file", str, 'train/trainer.py')]
    parser = argparse.ArgumentParser()
    for optional_arg, arg_type, arg_default in job_args:
        parser.add_argument(f"--{optional_arg}", help=f"{optional_arg}", default = arg_default, type=arg_type)

    for optional_arg, arg_type in optional_args:
        parser.add_argument(f"--{optional_arg}", help=f"{optional_arg}", type=arg_type)

    for optional_arg, arg_type in other_args:
        parser.add_argument(f"--{optional_arg}", help=f"{optional_arg}", type=arg_type)

    args = parser.parse_args()

    ## execute first command first
    trainer_cmd = f'srun -c {args.gpu_count * args.cpu_ratio}{" -w " + args.partition if len(args.partition) > 0 else ""} --gres=gpu:{args.gpu_count} -p {"cwfu_gpu" if args.use_fu_gpu else "gpu_24h"}{" --qos cwfu_gpu --account cwfu_gpu" if args.use_fu_gpu else ""} --pty python {args.run_file}'
    trainer_cmd_tmp = trainer_cmd + f' --special_symbol _{config.exp_idx if args.__dict__.get("exp_id", None) is None else args.__dict__.get("exp_id", None)}_{random_string}_{running_cnt}'
    for optional_arg, arg_type in optional_args:
        if args.__dict__.get(optional_arg, None) is not None:
            trainer_cmd_tmp = trainer_cmd_tmp + f' --{optional_arg} {args.__dict__.get(optional_arg, None)}'

    ### set exp id
    if args.__dict__.get('exp_id', None) is not None:
        config.exp_idx = args.__dict__.get('exp_id', None)

    print(f"executed cmd : {trainer_cmd_tmp}")
    process = subprocess.Popen(trainer_cmd_tmp.split(" "))
    print(f"process id : {process.pid}")
    process.communicate()

    ## get information from log
    # saved model
    input_dicts = {
        optional_arg: None if arg_type == str else -1 for optional_arg, arg_type in optional_args
    }


    while (input_dicts.get('starting_epoch', -1) < config.training_epochs - 1):
        previous_iter_folder = get_previous_iteration_folder(random_string)
        new_input_dicts = deepcopy(input_dicts)
        new_input_dicts = get_prev_iter_informations(previous_iter_folder, new_input_dicts)
        new_input_dicts['resume_path'] = os.path.join(config.debug_base_folder, previous_iter_folder, 'config.py')
        print(f"current epoch : {new_input_dicts.get('starting_epoch', -1)} previous epoch : {input_dicts.get('starting_epoch', -1)}")
        if new_input_dicts.get('starting_epoch', -1) == input_dicts.get('starting_epoch', -1):
            print("cannot run!...... Might have bugs!")
            break
        input_dicts = new_input_dicts

        ## update random string
        random_string = get_random_string(length)
        running_cnt += 1
        trainer_cmd_tmp = trainer_cmd + f' --special_symbol _{config.exp_idx}_{random_string}_{running_cnt}'
        for optional_arg, arg_type in optional_args:
            if input_dicts.get(optional_arg, None):
                trainer_cmd_tmp += f' --{optional_arg} {input_dicts.get(optional_arg)}'

        print(f"executed cmd : {trainer_cmd_tmp}")
        process = subprocess.Popen(trainer_cmd_tmp.split(" "))
        print(f"process id : {process.pid}")
        process.communicate()
