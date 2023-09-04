import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as TF


def transform(img_array, input_size):
    """

    :param img_array:
    :param input_size:
    :return:
    """
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_array)

    width, height = img.size
    img = TF.resize(img, int(height / width * input_size))  # the smaller edge will be matched to input_size
    img = TF.pad(img, (0, int((img.size[0] - img.size[1]) / 2)))

    tensor = TF.to_tensor(img)
    # tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return tensor


def stack_tensors(tensors):
    stacked = torch.stack(tensors)
    return stacked


def preds_postprocess(preds_output, preds_info, stream_names):
    preds_dict = {}
    cls_dict = {}

    for i, output in enumerate(preds_output):
        if output[0] is None:
            preds_dict[stream_names[i]] = None
            cls_dict[stream_names[i]] = None
        else:
            ratio = preds_info[i]["ratio"]

            cls = []
            person_bboxes = []
            pred = output[0].cpu()
            for item in pred:
                xyxy = item[0:4]
                xyxy /= ratio

                person_bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))  # convert tensor to int
                person_bboxes.append(person_bbox)
                cls.append(item[4])
            if len(person_bboxes) != 0:
                preds_dict[stream_names[i]] = person_bboxes
                cls_dict[stream_names[i]] = cls
            else:
                preds_dict[stream_names[i]] = None
                cls_dict[stream_names[i]] = None

    return preds_dict, cls_dict


