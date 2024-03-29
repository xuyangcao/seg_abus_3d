import numpy as np 
import os
import argparse 
import tqdm
import pandas as pd
import SimpleITK as sitk 
from medpy import metric

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='')
    parser.add_argument('--fold', type=int, default=5)

    args = parser.parse_args()
    # save csv file to the current folder
    if args.file_path[-1] == '/':
        args.save = args.file_path[:-1] + '.csv'
    else:
        args.save = args.file_path + '.csv'

    return args

def get_df(file_path):
    filenames = os.listdir(file_path)

    dsc_list = []
    jc_list = []
    hd_list = []
    hd95_list = []
    asd_list = []
    for filename in tqdm.tqdm(filenames):
        gt_img = sitk.ReadImage(os.path.join(file_path, filename+'/gt.nii.gz'))
        gt_volume = sitk.GetArrayFromImage(gt_img)

        pre_img = sitk.ReadImage(os.path.join(file_path, filename+'/pred.nii.gz'))
        pre_volume = sitk.GetArrayFromImage(pre_img)

        dsc = metric.binary.dc(pre_volume, gt_volume)
        jc = metric.binary.jc(pre_volume, gt_volume)
        if np.sum(pre_volume) == 0:
            print('prediction == 0')
            hd = 10
            hd95 = 10
            asd = 10
        else:
            hd = metric.binary.hd(pre_volume, gt_volume, voxelspacing=(0.4, 0.4, 0.4))
            hd95 = metric.binary.hd95(pre_volume, gt_volume, voxelspacing=(0.4, 0.4, 0.4))
            asd = metric.binary.asd(pre_volume, gt_volume, voxelspacing=(0.4, 0.4, 0.4))

        dsc_list.append(dsc)
        jc_list.append(jc)
        hd_list.append(hd)
        hd95_list.append(hd95)
        asd_list.append(asd)

    df = pd.DataFrame()
    df['name'] = filenames
    df['dsc'] = np.array(dsc_list)
    df['jc'] = np.array(jc_list) 
    df['hd'] = np.array(hd_list) 
    df['hd95'] = np.array(hd95_list) 
    df['asd'] = np.array(asd_list) 
    #print(df.describe())
    #df.to_csv(args.save)
    return df

def main():
    args = get_args()
    df_all = pd.DataFrame()
    for i in range(1, args.fold+1):
        file_path = args.file_path+str(i)
        df = get_df(file_path)
        df_all = df_all.append(df)
    df_all.to_csv(args.save)
    print(df_all.describe())


if __name__ == '__main__':
    main()
