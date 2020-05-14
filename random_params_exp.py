import json
import os
import subprocess
import uuid

import numpy as np
import yaml

PARAMETERS = {
    'gan_config.gen_hidden_size': [64, 256],
    'gan_config.dis_hidden_1_size': [128, 2048],
    'gan_config.dis_hidden_2_size': [128, 2048],
    'gan_config.dis_maxpool_reduction': [2, 16],
    'gan_config.gradient_penalty_ratio': [8, 8],
    'use_all_letters': [1, 2]
}


def how_many(x, y):
    i = 0
    if np.isclose(x, y):
        return i
    for i in range(50):
        if np.isclose(x, y):
            return i
        x *= 2
        i += 1
    return 0


def sample_values():
    values = {
        k: (2 ** np.random.randint(how_many(*v) + 1)) * (v[0])
        for k, v in PARAMETERS.items()
    }
    for k, v in values.items():
        assert PARAMETERS[k][0] <= v <= PARAMETERS[k][1]
    return values


def run_exp():
    params = sample_values()
    exp_id = str(uuid.uuid4())[:8]
    if params['use_all_letters'] == 1:
        params['gan_config.use_all_letters'] = True
        params['gan_config.max_sentence_length'] = 360
    else:
        params['gan_config.use_all_letters'] = False
        params['gan_config.max_sentence_length'] = 64
    del params['use_all_letters']
    params['Model.encoder.identity'] = True
    params['Trainer.num_epochs'] = 1

    for _try in range(2):
        destination_dir = (
            f'runs/2020_05_15/{exp_id}_all_letters_' \
            f'{params["gan_config.use_all_letters"]}/{_try}'
        )
        run_cmd = [
            './train.sh',
            'GAN_supervised_encoder.yaml',
            destination_dir,
            '--rng-seed', f'{np.random.randint(9999)}',
            # '--initialize-from', '55_sup_enc.pkl',
            # '-r', 'gan', 'probe'
            '-d'
        ]
        for k, v in params.items():
            run_cmd.extend(['-m', k, str(v)])
        subprocess.run(run_cmd)
        with open(os.path.join(
            os.path.dirname(destination_dir),
            'params.yaml'
        ), 'w') as f:
            yaml.dump(params, f)


if __name__ == '__main__':
    while True:
        run_exp()
