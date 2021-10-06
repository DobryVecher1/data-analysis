import numpy as np
import pandas as pd
#------------------------------------------
def ohe(df, features, drop_flag):
    """
    one-hot encoder.
    df -- DataFrame
    features -- лист из признаков, которые хотим закодировать
    drop_flag -- True, если хотим удалить исходные колонки признаков
    """
    for feat in features:
        categ_list = df[feat].unique()
        df_enc = np.zeros((df.shape[0], len(categ_list)))
        for ii in range(len(categ_list)):
            df_enc[:, ii] = (df[feat]==categ_list[ii]).astype(int) 

        df_enc = pd.DataFrame(data=df_enc,
                            index=df.index, 
                            columns=list(map(lambda elem: feat+'_'+elem, list(map(str, categ_list)))))
        df = pd.concat([df, df_enc], axis=1)
    if drop_flag:
        df.drop(columns=features, inplace=True)
        
    return df
#------------------------------------------
def num_heroes(df):
    """
    Возвращает количество персонажей
    """
    n_heroes = 0
    for ii in range(10):
        if ii <= 4:
            max_id = df['r'+str(ii+1)+'_hero'].max()
        else:
            max_id = df['d'+str(ii-4)+'_hero'].max()
    if max_id > n_heroes:
        n_heroes = max_id
    
    return n_heroes
#------------------------------------------
def heroes_bag(df, num_heroes):
    """
    Функция добавляет N признаков, при этом i-й будет равен нулю, если i-й герой не участвовал в матче; 
    единице, если i-й герой играл за команду Radiant; минус единице, 
    если i-й герой играл за команду Dire
    """
    df_pick = np.zeros((df.shape[0], num_heroes))
    for ii, match_id in enumerate(df.index):
        for jj in range(5):
            df_pick[ii, df['r%d_hero' % (jj+1)][match_id]-1] = 1
            df_pick[ii, df['d%d_hero' % (jj+1)][match_id]-1] = -1
    # Добавим новые признаки
    df_pick = pd.DataFrame(data=df_pick, 
                        index=df.index,
                        columns=list(map(lambda elem: 'hero_'+str(elem), list(range(1, num_heroes+1)))))
    df = pd.concat([df, df_pick], axis=1)

    return df