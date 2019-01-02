import sys
import os
import argparse
import subprocess

def main(opts):
  list_dir = os.listdir(opts.db_path)
  
  #print(list_dir)
  #print(len(list_dir))
  
  for i in range(len(list_dir)):
    img = list_dir[i]
    img =  opts.db_path + "/" + img
    print(img)
    subprocess.call(["python3","testone.py","--db_path",img])
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute testone.py to all images of a folder')

    
    parser.add_argument('--db_path', type=str, default='/veu4/usuaris27/pae2018/projects/Localitzacio/Marti/stage_1_test_images',
                        help='path to images folder')
    parser.add_argument('--out_path', type=str, default='output',
                        help='path to output folder')

    opts = parser.parse_args()
    main(opts)