import os
import io
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='')
    parser.add_argument('--output_file', default='')

    args = parser.parse_args()
    input_dir = args.input_dir
    output_file = args.output_file

    file_list = []
    dirs = os.listdir(input_dir)
    for each_dir in dirs:
        sub_dirs = os.listdir(os.path.join(input_dir, each_dir))
        for sub_dir in sub_dirs:
            files = os.listdir(os.path.join(input_dir, each_dir, sub_dir))
            for file_name in files:
                if file_name.endswith('.jpg'):
                    file_list.append(os.path.join(each_dir, sub_dir, file_name))

    save_file = open(output_file, 'w')
    for file_name in file_list:
        save_file.write('%s\n' % file_name)
