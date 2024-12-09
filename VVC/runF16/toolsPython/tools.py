# -*- coding: utf-8 -*-
# Program:
#       This script is used for analysis of the maneuvers data
# History:
#   2019/8/16   WH      created   
#   2019/8/16   WH      modified

import os
import re
import numpy as np

# os_walk_test
def os_walk_test(file_dir = None):
    if file_dir is None:
        file_dir = os.getcwd()
    '''
    os.walk Generate the file names in a directory tree by walking the tree either top-down or bottom-up.
    For each directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    '''
    for root, dirs, files in os.walk(file_dir):
        print(root)
        print(dirs)
        print(files)
        
''' return a list containing all files in the root directory: file_dir, with its absolute path '''
def get_all_files(file_dir = None):
    if file_dir is None:
        file_dir = os.getcwd()
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        file_list.extend([root+'/'+f for f in files])
    return file_list

''' get file names with the ext in the directory, if the root is True, all files will be return, including sub-folders'''
def get_file_names(ext = None, file_dir = None, root=False):
    if file_dir is None:
        file_dir = os.getcwd()
    if root:
        myFile = get_all_files(file_dir)
    else:
        myFile = [file_dir + '/' + f for f in os.listdir(file_dir)]
    if ext is None:
        return myFile
    else:
        return [file for file in myFile if file.endswith(ext)]

''' remove the letters specified in the str in the myList '''
def del_str_in_item_of_list(myList, str):
    return [re.sub(str, '', item) for item in myList]

def new_dir_path(dirname='results', path=None):
    # create new directory
    if path is None:
        path = './'
    new_dir_name = path + dirname
    if not(os.path.exists(new_dir_name) and os.path.isdir(new_dir_name)):
        os.makedirs(new_dir_name)
    return new_dir_name+'/'
    
if __name__ == '__main__':
    # dir = '.'
    os_walk_test()