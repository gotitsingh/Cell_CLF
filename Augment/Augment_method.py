import random
from Augment.Transformations import apply_augment,augment_list
class Augmentation(object):
    def __init__(self):

        self.policy_list=[fn.__name__ for fn, v1, v2 in augment_list()]

    def __call__(self, img):
        for _ in range(1):
            name=random.choice(self.policy_list)
            magnitude=random.random()
            img,param = apply_augment(img, name, magnitude)

        return img
