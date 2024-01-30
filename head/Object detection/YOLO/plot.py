import torch
import config
import os
import utils
from tqdm import tqdm
from data import YoloPascalVocDatatset
from models import *
from torch.utils.data import DataLoader


MODEL_DIR = 'model_dir'


def plot_test_images():
    classes = utils.load_class_array()

    dataset = YoloPascalVocDatatset('test', normalize=True, augment=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = Yolov1ResNet()
    model.eval()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'weights', 'final')))

    count = 0
    with torch.no_grad():
        for image, labels, originial in tqdm(loader):
            predictions = model.forward(image)
            for i in range(image.size(dim=0)):
                utils.plot_boxes(
                    original[i, :, :, :],
                    predictions[i, :, :, :],
                    classes,
                    file=os.path.join('results', f'{count}')
                )
                # utils.plot_boxes(
                #   original[i, :, :, :],
                #   labels[i, :, :, :],
                #   classes,
                #   color='green'
                # )
                count += 1


if __name__ == '__main__':
    plot_test_images()