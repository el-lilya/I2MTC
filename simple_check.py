import pandas as pd
from ds.load_data import urls_from_clip, imgs_from_url, get_data
import os
from torchvision import models
import clip
import numpy as np


def main():
    (print(clip.available_models()))
    model, preprocess = clip.load("RN50")
    # model.eval()
    # input_resolution = model.visual.input_resolution
    # context_length = model.context_length
    # vocab_size = model.vocab_size
    #
    # print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()])}")
    # print("Input resolution:", input_resolution)
    # print("Context length:", context_length)
    # print("Vocab size:", vocab_size)
    print(preprocess)
    descriptions = dict(enumerate(['empty slots', 'pepper plant', 'tomato plant', 'kohlrabi plant', 'frisee plant',
                                   'lettuce plant', 'mint plant', 'lettuce oakleaf plant', 'radish plant', 'basil plant',
                                   'curly parsley plant', 'cress plant', 'chard plant', 'brassica plant',
                                   'lettuce endivia plant', 'parsley plant', 'chives plant']))
    text_tokens = clip.tokenize(["This is " + desc for desc in descriptions])


if __name__ == "__main__":
    main()
