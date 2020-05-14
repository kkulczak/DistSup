import itertools
import json
import os
import subprocess
from time import sleep
import uuid

import numpy as np
import yaml

PARAMETERS = {
    'gan_config.dis_hidden_1_size': [2048, 2048 * 2],
    'gan_config.dis_hidden_2_size': [2048, 2048 * 2],
    'gan_config.dis_maxpool_reduction': [2, 32],
    'gan_config.gradient_penalty_ratio': [8, 8],
    'lr': [0.001, 0.001]
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
    params['gan_config.dis_learning_rate'] = params['lr'] * 2
    params['gan_config.gen_learning_rate'] = params['lr']
    del params['lr']
    exp_id = str(uuid.uuid4())[:12]
    for _try in range(8):
        run_cmd = [
            './train.sh',
            'GAN_supervised_encoder.yaml', f'runs/2020_05_05/{exp_id}/{_try}',
            '--rng-seed', f'{np.random.randint(9999)}',
            '--initialize-from', '55_sup_enc.pkl',
            '-r', 'gan', 'probe'
        ]
        for k, v in params.items():
            run_cmd.extend(['-m', k, str(v)])
        subprocess.run(run_cmd)
        with open(f'runs/2020_05_05/{exp_id}/params.yaml', 'w') as f:
            yaml.dump(params, f)


def run_id_exp():
    path = 'runs/2020_05_14/id_noise/'
    noises = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for noise, seed in zip(itertools.cycle(noises), itertools.count()):
        name = f'noise_{noise}_id_{seed}'
        run_cmd = [
            './train.sh',
            'GAN_supervised_encoder.yaml',
            os.path.join(path, name),
            '--rng-seed', f'{seed}',
            '--initialize-from', '55_sup_enc.pkl',
            '-r', 'gan', 'probe'
        ]
        params = {
            'gan_config.batch_inject_noise': noise,
            'Trainer.num_epochs': 10,
        }
        for k, v in params.items():
            run_cmd.extend(['-m', k, str(v)])
        subprocess.run(run_cmd)

if __name__ == '__main__':
    run_id_exp()
