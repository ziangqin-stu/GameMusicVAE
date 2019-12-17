from urllib.request import urlretrieve

import os


def save_midi(s_url_list, save_path):
    counter = 0
    for s_url in s_url_list:
        try:
            name = s_url.split("/")[-2] + "_" + s_url.split("/")[-1]
            urlretrieve(s_url, save_path + "/" + name)
            counter += 1
            print("saved " + str(counter) + " | " + name + "!")
        except BaseException:
            pass
        continue


def save_list(url_list, save_path, save_name):
    with open(os.path.join(save_path, save_name), 'w') as f:
        for line in url_list:
            f.write(str(line) + '\n')
    print("saved list at " + os.path.join(save_path, save_name) + "!")


def load_list(file_path_name):
    with open(file_path_name, 'r') as f:
        score = [line.rstrip('\n') for line in f]
    print("loaded list from " + file_path_name + "!")
    return score
