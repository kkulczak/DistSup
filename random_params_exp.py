from argparse import ArgumentParser
import json
import os
import random
import subprocess
import uuid

import numpy as np
import yaml

PARAMETERS = {
    # 'Model.letters_protos.protos_per_token': [1],
    # 'Model.letters_protos.preproc_softmax': [True],
    'Trainer.num_epochs': 40,
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


def sample(val):
    if isinstance(val, (int, float, bool)):
        return val
    if len(val) == 2 and all(
        (type(x) is int or type(x) is float)
        for x in val
    ):
        v = (2 ** np.random.randint(how_many(*val) + 1)) * (val[0])
        assert val[0] <= v <= val[1]
        return v
    return random.choice(val)


def sample_values():
    values = {
        k: sample(v)
        for k, v in PARAMETERS.items()
    }
    return values


def run_exp(dir_name, how_many=1, debug=False):
    params = sample_values()
    exp_id = str(uuid.uuid4())[:8]
    if 'use_all_letters' in params:
        if params['use_all_letters'] == 1:
            params['gan_config.use_all_letters'] = True
            params['gan_config.max_sentence_length'] = 384
        else:
            params['gan_config.use_all_letters'] = False
            params['gan_config.max_sentence_length'] = 64
        del params['use_all_letters']

    for _try in range(how_many):
        destination_dir = (
            f'{dir_name}'
            f'/{exp_id}'
            # f'_protos_per_token'
            # f'{params["Model.letters_protos.protos_per_token"]}'
            f'/{_try}'
        )
        run_cmd = [
            './train.sh',
            'GAN_supervised_encoder.yaml',
            destination_dir,
            '--rng-seed', f'{np.random.randint(9999)}',
            '--initialize-from', '55_sup_enc.pkl',
            '-r', 'gan', 'probe'
        ]
        if debug:
            run_cmd.append('-d')
            params['Trainer.num_epochs'] = 1
        for k, v in params.items():
            run_cmd.extend(['-m', k, str(v)])
        subprocess.run(run_cmd)
        with open(os.path.join(
            os.path.dirname(destination_dir),
            'params.yaml'
        ), 'w') as f:
            yaml.dump(params, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dir_name', type=str,
                        help='Directory where save runs results')
    parser.add_argument('--repeat', '-r', type=int, default=1,
                        help='how many runs for single args sample')
    parser.add_argument('-d', '--debug', action='store_true')

    args = parser.parse_args()
    while True:
        run_exp(dir_name=args.dir_name, how_many=args.repeat, debug=args.debug)
