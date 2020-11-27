
import numpy as np
import sys
import os
def Locate_Cell_Area(Segment_Array,params):
    sys.setrecursionlimit(100000)
    os.system("ulimit -s 100000")
    width = Segment_Array.shape[0]
    height = Segment_Array.shape[1]
    Locate_Array=np.zeros([width,height])
    #init location array
    for j in range(width):
        for k in range(height):
            Locate_Array[j,k]=j*height+k
            if Segment_Array[j,k]==-1:
                Locate_Array[j,k]=-1
    all_label = np.unique(Locate_Array)
    print("After clear the background, we segmented %d areas in this part" % len(all_label))

    for j in range(width):
        for k in range(height):
            if Locate_Array[j,k]==j*height+k:
                tmp_label=j*height+k
                Update_Locate_Array(Locate_Array,Segment_Array,j,k,tmp_label)
    #clear those include less than 10 pixels
    all_label=np.unique(Locate_Array)
    print("In total, we segmented %d areas in this part"%len(all_label))
    for tmp_label in all_label:
        tmp_coord=np.argwhere(Locate_Array==tmp_label)
        if len(tmp_coord)<=10:
            for tmp_tmp_coord in tmp_coord:
                Locate_Array[tmp_tmp_coord[0],tmp_tmp_coord[1]]=-2
    all_label = np.unique(Locate_Array)
    print("After removing some small parts, we segmented %d areas in this part" % len(all_label))
    for tmp_label in all_label:
        tmp_coord=np.argwhere(Locate_Array==tmp_label)
        if len(tmp_coord)>=4*params['width']*params['height']:
            for tmp_tmp_coord in tmp_coord:
                Locate_Array[tmp_tmp_coord[0],tmp_tmp_coord[1]]=-2
    all_label = np.unique(Locate_Array)
    print("After removing some big parts, we segmented %d areas in this part" % len(all_label))
    return Locate_Array


def Update_Locate_Array(Locate_Array,Segment_Array,x_coord,y_coord,tmp_label,verify=False):
    if x_coord<0 or y_coord<0 or x_coord>=Locate_Array.shape[0] or y_coord>=Locate_Array.shape[1]:
        return
    if verify and Locate_Array[x_coord,y_coord]==tmp_label:
        return
    if Segment_Array[x_coord,y_coord]!=1:
        if Segment_Array[x_coord,y_coord]!=-1:
            Locate_Array[x_coord,y_coord]=-2#unsure label
        return

    Locate_Array[x_coord,y_coord]=tmp_label
    if x_coord+1<Locate_Array.shape[0]:#to reduce the recursive calling
        if Segment_Array[x_coord+1,y_coord]==1:
            if Locate_Array[x_coord+1,y_coord]!=tmp_label:
                Update_Locate_Array(Locate_Array, Segment_Array, x_coord+1, y_coord, tmp_label,verify=True)
    if x_coord -1>=0:
        if Segment_Array[x_coord-1,y_coord]==1:
            if Locate_Array[x_coord-1,y_coord]!=tmp_label:
                Update_Locate_Array(Locate_Array, Segment_Array, x_coord - 1, y_coord, tmp_label,verify=True)
    if y_coord+1<Locate_Array.shape[1]:
        if Segment_Array[x_coord,y_coord+1]==1:
            if Locate_Array[x_coord,y_coord+1]!=tmp_label:
                Update_Locate_Array(Locate_Array, Segment_Array, x_coord, y_coord+1, tmp_label,verify=True)
    if y_coord -1>=0:
        if Segment_Array[x_coord,y_coord-1]==1:
            if Locate_Array[x_coord,y_coord-1]!=tmp_label:
                Update_Locate_Array(Locate_Array, Segment_Array, x_coord, y_coord-1, tmp_label,verify=True)
