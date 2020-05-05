import json
import subprocess
import uuid

import numpy as np
import yaml

PARAMETERS = {
    'gan_config.dis_hidden_1_size': [128, 2048],
    'gan_config.dis_hidden_2_size': [128, 2048],
    'gan_config.dis_maxpool_reduction': [2, 32],
    'gan_config.gradient_penalty_ratio': [0.5, 32],
    'lr': [0.0001, 0.0128]
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
    for _try in range(2):
        run_cmd = [
            './train.sh',
            'GAN_supervised_encoder.yaml', f'runs/2020_05_05/{exp_id}/{_try}',
            '--rng-seed', f'{100 + _try}',
            '--initialize-from', '55_sup_enc.pkl',
            '-r', 'gan', 'probe'
        ]
        for k, v in params.items():
            run_cmd.extend(['-m', k, str(v)])
        print(' '.join(run_cmd))
        subprocess.run(run_cmd)
        with open(f'runs/2020_05_05/{exp_id}/params.yaml', 'w') as f:
            yaml.dump(params, f)


if __name__ == '__main__':
    while True:
        run_exp()
