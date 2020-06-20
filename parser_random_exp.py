from glob import glob
import itertools
import os

from dask import bag, diagnostics
import pandas as pd
from tqdm import tqdm
import yaml
import csv


def parse_results(EXP_DIR='runs/2020_05_05'):
    exps_params = glob(os.path.join(EXP_DIR, '*', 'params.yaml'))
    data_params = {
        path.split('/')[-2]: yaml.safe_load(open(path))
        for path in exps_params
    }
    df = pd.DataFrame.from_dict(data_params, orient='index')
    df = df.rename(columns=lambda x: x.replace('gan_config.', ''))
    if 'gen_learning_rate' in df:
        df['lr'] = df['gen_learning_rate']
        df = df.drop(columns=['dis_learning_rate', 'gen_learning_rate'])
    df: pd.DataFrame = df.rename(columns=lambda x: x.replace('dis_', ''))
    experiments_params = df
    columns = ['exp_id', '_try', 'name', 'step', 'value']

    def f(x):
        data = []
        tries_files = glob(
            os.path.join(EXP_DIR, x, '*', 'dev', 'events*.csv')
        )
        for p in tries_files:
            _id = p.split('/')[-3]
            df = pd.read_csv(p)
            df['exp_id'] = x
            df['_try'] = _id
            df = df[columns]
            data.append(df)
        return data

    with diagnostics.ProgressBar():
        dfs = bag.from_sequence(experiments_params.index).map(f).compute()

    return pd.concat(itertools.chain(*dfs)), experiments_params


def main():
    data, params = parse_results()
    print(params.head())
    print(data.head())


if __name__ == '__main__':
    main()
