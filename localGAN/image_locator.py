import os
import numpy as np
import pandas as pd
import argparse


def get_locs(top_folder, out_csv, classes):
    
    if type(classes) is not list:
        classes = [int(classx) for classx in classes.split(',')]
        
    classes_loc_dict = {}
    for folder in os.listdir(top_folder):
        if int(folder.split('_')[1]) in classes:
            out_list = []
            for file in os.listdir(os.path.join(top_folder,folder)):
                lon = float(file[file.find('lon')+4:file.find('lat')-2].replace("neg","-").replace("_","."))
                lat = float(file[file.find('lat')+4:file.find('.')].replace("neg","-").replace("_","."))
                out_list.append([lon,lat])
            classes_loc_dict[folder] = out_list

    dfs_list = [pd.DataFrame([[item[0], item[1]] for item in pd.DataFrame.from_dict(classes_loc_dict[class_x]).to_numpy()]) for class_x in classes_loc_dict.keys()]
    output_df = pd.concat(dfs_list,ignore_index=True)
    output_df.index.name = 'OBJECTID'
    output_df.columns = ['x', 'y']
    output_df.to_csv(out_csv)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate xy locs from file names.')
    parser.add_argument("--top_folder",
                        help='folder containing all class folders.')
    parser.add_argument("--out_csv",
                        help='name of output csv',
                        default="class_locations")
    parser.add_argument("--classes",
                        help='comma-separated list of class numbers')
    args = parser.parse_args()
    get_locs(args.in_platform, args.out_size, args.out_format, args.multispectral)