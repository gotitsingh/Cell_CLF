import numpy as np
from collections import defaultdict
def Find_Segment_Area(im_target_rgb):
    width=im_target_rgb.shape[0]
    height=im_target_rgb.shape[1]
    Color_Dict=defaultdict(list)
    for j in range(width):
        for k in range(height):
            R_color=int(im_target_rgb[j,k,0])
            G_color = int(im_target_rgb[j, k, 1])
            B_color = int(im_target_rgb[j, k, 2])
            RGB_key=str(R_color)+','+str(G_color)+','+str(B_color)
            coordinate_info=str(j)+','+str(k)
            Color_Dict[RGB_key].append(coordinate_info)
    all_possible_coord=width*height
    minimum_coord=0.05*all_possible_coord#give up those area where 2nd large covering is smaller than this
    Segment_array=np.zeros(im_target_rgb.shape[:2])
    #first label all those largest color as background
    max_number=0
    max_key=-1
    for key in Color_Dict:
        tmp_coord=Color_Dict[key]
        if len(tmp_coord)>max_number:
            max_number=len(tmp_coord)
            max_key=key
    assert max_key!=-1
    background_coord=Color_Dict[max_key]
    for tmp_coord in background_coord:
        tmp_split_coord=tmp_coord.split(",")
        tmp_x=int(tmp_split_coord[0])
        tmp_y=int(tmp_split_coord[1])
        Segment_array[tmp_x,tmp_y]=-1
    print("We have %d background pixels "%len(background_coord))
    del Color_Dict[max_key]
    #2nd label all the cells in the coord
    max_number = 0
    max_key = -1
    for key in Color_Dict:
        tmp_coord = Color_Dict[key]
        if len(tmp_coord) > max_number:
            max_number = len(tmp_coord)
            max_key = key
    if max_key==-1:
        return Segment_array

    cell_coord = Color_Dict[max_key]
    print("We have %d cell pixels"%len(cell_coord))
    if len(cell_coord)<=minimum_coord:
        print("Because the pixels are too small, we give up to study it")
        return Segment_array
    for tmp_coord in cell_coord:
        tmp_split_coord = tmp_coord.split(",")
        tmp_x = int(tmp_split_coord[0])
        tmp_y = int(tmp_split_coord[1])
        Segment_array[tmp_x, tmp_y] = 1
    del Color_Dict[max_key]
    max_number = 0
    max_key = -1
    for key in Color_Dict:
        tmp_coord = Color_Dict[key]
        if len(tmp_coord) > max_number:
            max_number = len(tmp_coord)
            max_key = key
    if max_key == -1:
        return Segment_array

    cell_coord = Color_Dict[max_key]
    print("We have %d cell pixels" % len(cell_coord))
    if len(cell_coord) <= minimum_coord:
        print("2nd adding,Because the pixels are too small, we give up to study it")
        return Segment_array
    for tmp_coord in cell_coord:
        tmp_split_coord = tmp_coord.split(",")
        tmp_x = int(tmp_split_coord[0])
        tmp_y = int(tmp_split_coord[1])
        Segment_array[tmp_x, tmp_y] = 1
    return Segment_array


