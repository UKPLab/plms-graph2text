import re
import os

def convert_text(text):
    #return text
    text = text.lower()
    text = ' '.join(re.split('(\W)', text))
    text = ' '.join(text.split())
    return text

def eval_meteor_test_webnlg(folder_data, pred_file, dataset):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"

    cmd_string = "java -jar " + folder_data_before + "/meteor-1.5.jar " + pred_file + " " \
                  + folder_data + "/" + dataset + ".target_eval_meteor -l en -norm -r 3 > " + pred_file.replace("txt", "meteor")

    os.system(cmd_string)

    meteor_info = open(pred_file.replace("txt", "meteor"), 'r').readlines()[-1].strip()

    return meteor_info


def eval_chrf_test_webnlg(folder_data, pred_file, dataset):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"

    cmd_string = "python " + folder_data_before + "/chrf++.py -H " + pred_file + " -R " \
                  + folder_data + "/" + dataset + ".target_eval_crf > " + pred_file.replace("txt", "chrf")

    os.system(cmd_string)

    chrf_info_1 = open(pred_file.replace("txt", "chrf"), 'r').readlines()[1].strip()
    chrf_info_2 = open(pred_file.replace("txt", "chrf"), 'r').readlines()[2].strip()

    return chrf_info_1 + " " + chrf_info_2


def eval_bleu(pred_file, folder_data, dataset):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"

    if dataset == 'val':
        dataset = 'dev'

    cmd_string = "perl " + folder_data_before + "/multi-bleu.perl -lc " + folder_data + "/" + dataset + ".target.tok"\
                 + " < " + pred_file + " > " + pred_file.replace("txt", "bleu_data")
    os.system(cmd_string)

    try:
        bleu_info_data = open(pred_file.replace("txt", "bleu_data"), 'r').readlines()[0].strip()
    except:
        bleu_info_data = 'no data'

    return bleu_info_data


def eval_meteor(ref_file, pred_file):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"

    cmd_string = "java -jar " + folder_data_before + "/meteor-1.5.jar " + pred_file + " " \
                  + ref_file + " > " + pred_file.replace("txt", "meteor")

    os.system(cmd_string)

    meteor_info = open(pred_file.replace("txt", "meteor"), 'r').readlines()[-1].strip()

    return meteor_info


def eval_chrf(ref_file, pred_file):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    folder_data_before = dir_path + "/../utils"

    cmd_string = "python " + folder_data_before + "/chrf++.py -H " + pred_file + " -R " \
                  + ref_file + " > " + pred_file.replace("txt", "chrf")

    os.system(cmd_string)

    try:
        chrf_info_1 = open(pred_file.replace("txt", "chrf"), 'r').readlines()[1].strip()
        chrf_info_2 = open(pred_file.replace("txt", "chrf"), 'r').readlines()[2].strip()
        chrf_data = chrf_info_1 + " " + chrf_info_2
    except:
        chrf_data = "no data"


    return chrf_data