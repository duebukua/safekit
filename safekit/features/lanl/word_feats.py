# -*- coding: UTF-8 -*-

"""
@authors: Ryan Baerwolf, Aaron Tuor (rdbaerwolf@gmail.com, baerwor@wwu.edu, aaron.tuor@pnnl.gov)

Derives word-level features for LANL using auth_h.txt. Weekend days are filtered out of data set.
"""

import argparse
import operator
import json

weekend_days = [3, 4, 10, 11, 17, 18, 24, 25, 31, 32, 38, 39, 45, 46, 47, 52, 53]
OOV_CUTOFF = 40

# Frequency Dicts
usr_counts = {}
pc_counts = {}
domain_counts = {}

sos = 0
eos = 1
# Lookups
usr_OOV = 2
pc_OOV = 3
domain_OOV = 4

usr_inds = {}
pc_inds = {}
domain_inds = {}

auth_dict = {}
logon_dict = {}
orient_dict = {}
success_dict = {}

curr_ind = 5  # All IDs should be unique across all type dictionaries, we want multiple types of OOVs


token_word_map = {}  # 所有token到word的映射都存在这里面
word_token_inds = {}  # 用于测试集生成token的倒排表,


def return_parser():
    """Configures and returns argparse ArgumentParser object for command line script."""
    parser = argparse.ArgumentParser("Convert raw loglines to ASCII code (minus 30) integer representation.")
    parser.add_argument('-datafile',
                        type=str,
                        default="/home/hutch_research/data/lanl/auth_h.txt",
                        help='Path to files to transliterate.')
    parser.add_argument('-outfile',
                        type=str,
                        help='Where to write derived features.')
    parser.add_argument('-record_dir',
                        type=str,
                        help='Directory to dump frequency counts, and word token mappings in')
    parser.add_argument('-redfile',
                        type=str,
                        help='Location of the completely specified redteam events file.')
    # 增加一个参数，测试集word到token的映射 直接用训练集的word_token_map.json 来生成
    parser.add_argument('-test',
                        type=int,
                        default=0,
                        help='generate test dataset.')
    return parser


def lookup(word, ind_dict, count_dict):
    """

    :param word: Raw text word token
    :param ind_dict: (dict) keys: raw word tokens, values: Integer representation
    :param count_dict: (dict) keys: raw word tokens, values: Number of occurrences
    :return: Integer representation of word
    """
    global curr_ind
    if count_dict is not None and (word in count_dict and count_dict[word] < OOV_CUTOFF):
    # if count_dict is not None and count_dict[word] < OOV_CUTOFF:
        if count_dict is usr_counts:
            return usr_OOV
        elif count_dict is pc_counts:
            return pc_OOV
        elif count_dict is domain_counts:
            return domain_OOV
    else:
        if word not in ind_dict:
            ind_dict[word] = curr_ind
            curr_ind += 1
        return ind_dict[word]


def increment_freq(ind_dict, key):
    """
    Used during -make_counts to track the frequencies of each element

    :param ind_dict: (dict) keys: Raw word token, values: integer representation
    :param key: Raw word token
    """
    if key in ind_dict:
        ind_dict[key] += 1
    else:
        ind_dict[key] = 1


def split_line(string):
    """
    Turn raw some fields of raw log line from auth_h.txt into a list of word tokens
    (needed for consistent user ids and domain ids)

    :param string: Raw log line from auth_h.txt
    :return: (list) word tokens for some fields of auth_h.txt
    """
    data = string.strip().split(',')
    src_user = data[1].split("@")[0]
    src_domain = data[1].split("@")[1]
    dst_user = data[2].split("@")[0]
    dst_domain = data[2].split("@")[1]
    src_pc = data[3]
    dst_pc = data[4]
    return src_user, src_domain, dst_user.replace("$", ""), dst_domain, src_pc, dst_pc


def get_line_counts(string):
    """
    Increments frequency counts for each element in a log line

    :param string: Raw log line from auth_h.txt
    """
    data = string.strip().split(",")
    # if len(data) != 9:
    if len(data) != 10:
        return

    src_user, src_domain, dst_user, dst_domain, src_pc, dst_pc = split_line(string)

    increment_freq(usr_counts, src_user)
    increment_freq(domain_counts, src_domain)
    increment_freq(domain_counts, dst_domain)
    if dst_user.startswith("U"):
        increment_freq(usr_counts, dst_user)
    else:
        increment_freq(pc_counts, dst_user)
    increment_freq(pc_counts, dst_pc)
    increment_freq(pc_counts, src_pc)


def translate_line(string):
    """
    Translates raw log line into sequence of integer representations for word tokens with sos and eos tokens.
    :param string: Raw log line from auth_h.txt
    :return: (list) Sequence of integer representations for word tokens with sos and eos tokens.
    """
    data = string.split(",")
    time = int(data[0])  # could be used to make categorical variables for day of week and time of day.

    src_user, src_domain, dst_user, dst_domain, src_pc, dst_pc = split_line(string)
    src_user = lookup(src_user, usr_inds, None)
    src_domain = lookup(src_domain, domain_inds, domain_counts)

    if dst_user.startswith('U'):
        dst_user = lookup(dst_user, usr_inds, None)
    else:
        dst_user = lookup(dst_user, pc_inds, pc_counts)
    dst_domain = lookup(dst_domain, domain_inds, domain_counts)

    src_pc = lookup(src_pc, pc_inds, pc_counts)
    dst_pc = lookup(dst_pc, pc_inds, pc_counts)

    if data[5].startswith("MICROSOFT_AUTH"):  # Deals with file corruption for this value.
        data[5] = "MICROSOFT_AUTH"
    auth_type = lookup(data[5], auth_dict, None)
    logon_type = lookup(data[6], logon_dict, None)
    auth_orient = lookup(data[7], orient_dict, None)
    success = lookup(data[8].strip(), success_dict, None)

    return "%s %s %s %s %s %s %s %s %s %s %s %s\n" % (str(sos), src_user, src_domain, dst_user,
                                                      dst_domain, src_pc, dst_pc, auth_type,
                                                      logon_type, auth_orient, success, str(eos))


def write_sorted_counts(count_dict, out_fn):
    """
    Sorts all of the elements in a dictionary by their counts and writes them to json and plain text
    :param count_dict: (dict) keys: word tokens, values: number of occurrences
    :param out_fn: (str) Where to write .json and .txt files to (extensions are appended)
    """
    sorted_counts = sorted(count_dict.items(), key=operator.itemgetter(1))
    json.dump(count_dict, open(out_fn + ".json", 'w'))
    with open(out_fn + ".txt", 'w') as outfile:
        for key, value in sorted_counts:
            outfile.write("%s, %s\n" % (key, value))


def second_to_day(seconds):
    """

    :param seconds: (int) Time in seconds starting at 0 as start of data collection.
    :return: (int) Time in days starting at 0 as start of data collection
    """

    return int(seconds) / 86400

def translate_line_test(string):
    """
    Translates raw log line into sequence of integer representations for word tokens with sos and eos tokens.
    :param string: Raw log line from auth_h.txt
    :return: (list) Sequence of integer representations for word tokens with sos and eos tokens.
    """
    data = string.split(",")
    time = int(data[0])  # could be used to make categorical variables for day of week and time of day.

    src_user, src_domain, dst_user, dst_domain, src_pc, dst_pc = split_line(string)
    src_user = lookup(src_user, word_token_inds, None)
    src_domain = lookup(src_domain, word_token_inds, domain_counts)

    if dst_user.startswith('U'):
        dst_user = lookup(dst_user, word_token_inds, None)
    else:
        dst_user = lookup(dst_user, word_token_inds, pc_counts)
    dst_domain = lookup(dst_domain, word_token_inds, domain_counts)

    src_pc = lookup(src_pc, word_token_inds, pc_counts)
    dst_pc = lookup(dst_pc, word_token_inds, pc_counts)

    if data[5].startswith("MICROSOFT_AUTH"):  # Deals with file corruption for this value.
        data[5] = "MICROSOFT_AUTH"
    auth_type = lookup(data[5], word_token_inds, None)
    logon_type = lookup(data[6], word_token_inds, None)
    auth_orient = lookup(data[7], word_token_inds, None)
    success = lookup(data[8].strip(), word_token_inds, None)

    return "%s %s %s %s %s %s %s %s %s %s %s %s\n" % (str(sos), src_user, src_domain, dst_user,
                                                      dst_domain, src_pc, dst_pc, auth_type,
                                                      logon_type, auth_orient, success, str(eos))


if __name__ == '__main__':

    args = return_parser().parse_args()
    if args.record_dir and not args.record_dir.endswith('/'):
        args.record_dir += '/'

    if args.test == 0:

        # pass to get token counts
        with open(args.datafile, 'r') as infile:
            infile.readline()
            for line_num, line in enumerate(infile):
                if line_num == 0:
                    continue
                if line_num % 100000 == 0:
                    print line_num
                linevec = line.strip().split(',')
                user = linevec[1]
                day = second_to_day(int(linevec[0]))
                if user.startswith('U') and day not in weekend_days:
                    get_line_counts(line)   # 频率计算
        write_sorted_counts(usr_counts, args.record_dir + "usr_counts")
        write_sorted_counts(pc_counts, args.record_dir + "pc_counts")
        write_sorted_counts(domain_counts, args.record_dir + "domain_counts")

        # pass to make features
        # with open(args.redfile, 'r') as red:
        #     redevents = set(red.readlines())

        with open(args.datafile, 'r') as infile, open(args.outfile, 'w') as outfile:
            # outfile.write(
            #     'line_number second day user red src_usr src_domain dst_usr dst_domain src_pc dst_pc auth_type logon auth_orient success\n')
            infile.readline()
            for line_num, line in enumerate(infile):
                if line_num % 10000 == 0:
                    print line_num
                raw_line = line.split(",")
                # if len(raw_line) != 9:
                if len(raw_line) != 10:
                    print('bad length')
                    continue
                sec = raw_line[0]
                user = raw_line[1].strip().split('@')[0]
                day = second_to_day(int(sec))
                # red = 0
                # red += int(line in redevents)
                red = raw_line[-1].strip('\n').strip('\r')
                if user.startswith('U') and day not in weekend_days:
                    outfile.write("%s %s %s %s %s %s" % (line_num,
                                                         sec,
                                                         day,
                                                         user.replace("U", ""),
                                                         red,
                                                         translate_line(line)))
        with open(args.record_dir + str(OOV_CUTOFF) + "_em_size.txt", 'w') as emsize_file:
            emsize_file.write("%s" % curr_ind)
        other_inds = {'sos': 0, 'eos': 1, 'usr_OOV': 2, 'pc_OOV': 3, 'domain_OOV': 4}

        for map, file in zip([usr_inds.items(),
                              pc_inds.items(),
                              domain_inds.items(),
                              auth_dict.items(),
                              logon_dict.items(),
                              orient_dict.items(),
                              success_dict.items(),
                              other_inds.items()],
                             ['usr_map.json',
                              'pc_map.json',
                              'domain_map.json',
                              'auth_map.json',
                              'logon_map.json',
                              'orient_map.json',
                              'success_map.json',
                              'other_map.json']):
            json.dump(map, open(args.record_dir + file, 'w'))

        b_usr_inds = {v: k for k, v in usr_inds.items()}      # 倒排表查询，字母——>数字
        b_pc_inds = {v: k for k, v in pc_inds.items()}
        b_domain_inds = {v: k for k, v in domain_inds.items()}
        b_auth_inds = {v: k for k, v in auth_dict.items()}
        b_logon_inds = {v: k for k, v in logon_dict.items()}
        b_orient_inds = {v: k for k, v in orient_dict.items()}
        b_success_inds = {v: k for k, v in success_dict.items()}
        b_other_inds = {v: k for k, v in other_inds.items()}

        back_mappings = dict(b_usr_inds.items() +
                             b_pc_inds.items() +
                             b_domain_inds.items() +
                             b_auth_inds.items() +
                             b_logon_inds.items() +
                             b_orient_inds.items() +
                             b_success_inds.items() +
                             b_other_inds.items())

        json.dump(back_mappings, open(args.record_dir + 'word_token_map.json', 'w'))    # 最终的映射表

    if args.test == 1:
        with open('/home/chengy/MyCode/safekit/my_exp/freq_count/usr_map.json', 'r') as usr_inds_f:
            usr_map = json.load(usr_inds_f)
        with open('/home/chengy/MyCode/safekit/my_exp/freq_count/pc_map.json', 'r') as pc_inds_f:
            pc_map = json.load(pc_inds_f)
        with open('/home/chengy/MyCode/safekit/my_exp/freq_count/domain_map.json', 'r') as domain_inds_f:
            domain_map = json.load(domain_inds_f)
        with open('/home/chengy/MyCode/safekit/my_exp/freq_count/auth_map.json', 'r') as auth_dict_f:
            auth_map = json.load(auth_dict_f)
        with open('/home/chengy/MyCode/safekit/my_exp/freq_count/logon_map.json', 'r') as logon_dict_f:
            logon_map = json.load(logon_dict_f)
        with open('/home/chengy/MyCode/safekit/my_exp/freq_count/orient_map.json', 'r') as orient_dict_f:
            orient_map = json.load(orient_dict_f)
        with open('/home/chengy/MyCode/safekit/my_exp/freq_count/success_map.json', 'r') as success_dict_f:
            success_map = json.load(success_dict_f)
        with open('/home/chengy/MyCode/safekit/my_exp/freq_count/other_map.json', 'r') as other_inds_f:
            other_map = json.load(other_inds_f)

        usr_inds = {l[0] : l[1] for l in usr_map}
        pc_inds = {l[0] : l[1] for l in pc_map}
        domain_inds = {l[0] : l[1] for l in domain_map}
        auth_dict = {l[0] : l[1] for l in auth_map}
        logon_dict = {l[0] : l[1] for l in logon_map}
        orient_dict = {l[0] : l[1] for l in orient_map}
        success_dict = {l[0] : l[1] for l in success_map}
        other_inds = {l[0] : l[1] for l in other_map}


        with open('/home/chengy/MyCode/safekit/my_exp/freq_count/usr_counts.json', 'r') as usr_counts_f:
            usr_counts = json.load(usr_counts_f)
        with open('/home/chengy/MyCode/safekit/my_exp/freq_count/pc_counts.json', 'r') as pc_counts_f:
            pc_counts = json.load(pc_counts_f)
        with open('/home/chengy/MyCode/safekit/my_exp/freq_count/domain_counts.json', 'r') as domain_counts_f:
            domain_counts = json.load(domain_counts_f)

        # word_token_inds = {v: k for k, v in token_word_map.items()}

        with open(args.datafile, 'r') as infile, open(args.outfile, 'w') as outfile:
            # outfile.write(
            #     'line_number second day user red src_usr src_domain dst_usr dst_domain src_pc dst_pc auth_type logon auth_orient success\n')
            infile.readline()
            for line_num, line in enumerate(infile):
                if line_num % 10000 == 0:
                    print line_num
                raw_line = line.split(",")
                # if len(raw_line) != 9:
                if len(raw_line) != 10:
                    print('bad length')
                    continue
                sec = raw_line[0]
                user = raw_line[1].strip().split('@')[0]
                day = second_to_day(int(sec))
                # red = 0
                # red += int(line in redevents)
                red = raw_line[-1].strip('\n').strip('\r')
                if user.startswith('U') and day not in weekend_days:
                    outfile.write("%s %s %s %s %s %s" % (line_num,
                                                         sec,
                                                         day,
                                                         user.replace("U", ""),
                                                         red,
                                                         translate_line(line)))

