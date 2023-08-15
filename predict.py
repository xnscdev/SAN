import cv2
import torch

from utils import load_config, load_checkpoint
from infer.Backbone import Backbone
from dataset import Words

params = load_config('14.yaml')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device

words = Words(params['word_path'])
params['word_num'] = len(words)
params['struct_num'] = 7
params['words'] = words

model = Backbone(params)
model = model.to(device)

load_checkpoint(model, None, params['checkpoint'])

model.eval()


def convert(nodeid, gtd_list):
    isparent = False
    child_list = []
    for i in range(len(gtd_list)):
        if gtd_list[i][2] == nodeid:
            isparent = True
            child_list.append([gtd_list[i][0], gtd_list[i][1], gtd_list[i][3]])
    if not isparent:
        return [gtd_list[nodeid][0]]
    else:
        if gtd_list[nodeid][0] == '\\frac':
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] == 'Above':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Below':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Right':
                    return_string += convert(child_list[i][1], gtd_list)
            for i in range(len(child_list)):
                if child_list[i][2] not in ['Right', 'Above', 'Below']:
                    return_string += ['illegal']
        else:
            return_string = [gtd_list[nodeid][0]]
            for i in range(len(child_list)):
                if child_list[i][2] in ['l_sup']:
                    return_string += ['['] + convert(child_list[i][1], gtd_list) + [']']
            for i in range(len(child_list)):
                if child_list[i][2] == 'Inside':
                    return_string += ['{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sub', 'Below']:
                    return_string += ['_', '{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Sup', 'Above']:
                    return_string += ['^', '{'] + convert(child_list[i][1], gtd_list) + ['}']
            for i in range(len(child_list)):
                if child_list[i][2] in ['Right']:
                    return_string += convert(child_list[i][1], gtd_list)
        return return_string


def run()
with torch.no_grad():
    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = torch.Tensor(img) / 255
    image = image.unsqueeze(0).unsqueeze(0)

    image_mask = torch.ones(image.shape)
    image, image_mask = image.to(device), image_mask.to(device)

    prediction = model(image, image_mask)

    latex_list = convert(1, prediction)
    latex_string = ' '.join(latex_list)
    print(latex_string)
