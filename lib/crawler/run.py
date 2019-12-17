from lib.crawler.config import *
from lib.crawler.data_storage import *
from lib.crawler.html_parser import *

def spider_man_mdidi(reuse=False, file_path_name="", save_path="./data"):
    """
    DFS traverse webpages with cutoff and download midi files to specified path
    :param reuse: use a persistanced url-list file to down load midi files
    :param file_path_name: path and file name of url-list file
    :save_path: the path of saving midi files
    """
    if not reuse:
        # initialize
        s_url_list = []
        new_url_list = []
        visited_url_list = []
        new_url_list.append(ROOT_URL)
        # dfs
        iter = 1
        while len(new_url_list) > 0:
            print("iter:" + str(iter) + "  remain url number:" + str(len(new_url_list)) + ", source# = " + str(len(s_url_list)))
            iter += 1
            url = new_url_list[0]
            new_url_list.remove(url)

            if url not in visited_url_list:
                visited_url_list.append(url)
                if (url.count("/") - ROOT_URL.count("/")) <= CUTOFF:
                    cur_s_url_list = get_new_data(url, ".mid")
                    s_url_list += [s_url for s_url in cur_s_url_list if s_url not in s_url_list]
                    if (url.count("/") - ROOT_URL.count("/")) < CUTOFF:
                        cur_url_list = get_new_urls(url, next_layer=False)
                        new_url_list += [url for url in cur_url_list if url not in new_url_list]
        # save
        save_list(s_url_list, save_path, "s_url_list.txt")
        save_midi(s_url_list, save_path)
    else:
        s_url_list = load_list(file_path_name)
        save_midi(s_url_list, save_path)


def spider_man_vgmusic(reuse=False, file_path_name="", save_path="./data"):
    """
    DFS traverse webpages with cutoff and download midi files to specified path
    :param reuse: use a persistanced url-list file to down load midi files
    :param file_path_name: path and file name of url-list file
    :save_path: the path of saving midi files
    """
    if not reuse:
        # initialize
        s_url_list = []
        new_url_list = []
        visited_url_list = []
        new_url_list.append(ROOT_URL)
        # dfs
        iter = 1
        while len(new_url_list) > 0:
            print("iter:" + str(iter) + "  remain url number:" + str(len(new_url_list)) + ", source# = " + str(len(s_url_list)))
            iter += 1
            url = new_url_list[0]
            new_url_list.remove(url)

            if url not in visited_url_list:
                visited_url_list.append(url)
                if (url.count("/") - ROOT_URL.count("/")) <= CUTOFF:
                    cur_s_url_list = get_new_data_vgmusic(url, ".mid")
                    s_url_list += [s_url for s_url in cur_s_url_list if s_url not in s_url_list]
                    if (url.count("/") - ROOT_URL.count("/")) < CUTOFF:
                        cur_url_list = get_new_urls_vgmusic(url)
                        new_url_list += [url for url in cur_url_list if url not in new_url_list]
        # save
        save_list(s_url_list, save_path, "s_url_list.txt")
        save_midi(s_url_list, save_path)
    else:
        s_url_list = load_list(file_path_name)
        save_midi(s_url_list, save_path)


# spider_man_mdidi(True, file_path_name="./data/s_url_list.txt")
# spider_man_mdidi(False)
# spider_man_vgmusic(True, file_path_name="./data/s_url_list.txt", save_path="./data")

if __name__ == '__main__':
    save_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), 'source', 'data')
    spider_man_vgmusic(False, file_path_name="./data/s_url_list.txt", save_path=save_path)