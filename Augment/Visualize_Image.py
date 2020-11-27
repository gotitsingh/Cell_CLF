import os
import random
from Augment.Transformations import apply_augment,augment_list
from ops.os_operation import mkdir
from PIL import Image, PILLOW_VERSION, ImageEnhance
def Visualize_Image(image_path):
    save_path=os.path.join(os.getcwd(),'Visualize_Example')
    mkdir(save_path)
    img=Image.open(image_path)
    policy_list = [fn.__name__ for fn, v1, v2 in augment_list()]
    for k in range(len(policy_list)):
        name = policy_list[k]
        magnitude = random.random()
        imgt, param = apply_augment(img, name, magnitude)
        tmp_img_path=os.path.join(save_path,str(name)+'_mag_'+str(magnitude)+'.jpg')
        imgt.save(tmp_img_path)


