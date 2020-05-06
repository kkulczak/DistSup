from glob import glob
import os

import pandas as pd
import yaml
import csv

EXP_DIR = 'runs/2020_05_05'


def parse_results():
    exps_params = glob(os.path.join(EXP_DIR, '*', 'params.yaml'))
    data_params = {
        path.split('/')[-2]: yaml.safe_load(open(path))
        for path in exps_params
    }
    df = pd.DataFrame.from_dict(data_params, orient='index')
    df = df.rename(columns=lambda x: x.replace('gan_config.', ''))
    df['lr'] = df['gen_learning_rate']
    df = df.drop(columns=['dis_learning_rate', 'gen_learning_rate'])
    df: pd.DataFrame = df.rename(columns=lambda x: x.replace('dis_', ''))
    experiments_params = df
    data: pd.DataFrame = pd.DataFrame(columns=['exp_id', '_try', 'name', 'step', 'value'])
    for x in experiments_params.index:
        tries_files = glob(
            os.path.join(EXP_DIR, x, '[0-9]', 'dev', 'events*.csv'))
        for p in tries_files:
            _id = p.split('/')[-3]
            df = pd.read_csv(p)
            df['exp_id'] = x
            df['_try'] = _id
            df = df[data.columns.tolist()]
            data = data.append(df)
    print(data.head())
    return data, experiments_params

def main():
    data, params = parse_results()
    print(params.head())
    print(data.head())

if __name__ == '__main__':
    main()
