import os
import os.path as osp
import cv2

import numpy as np
from copy import deepcopy

from shapely import geometry

img_dir = osp.expanduser('~/Data/Image/beauty/')

win_name = 'liuyifei'
img_name = osp.join(img_dir, 'liyifei.png')
# img_name = osp.join('zhaoliying.png')
# img_name = osp.join('image/none_1.png')



class Status:
    PAINT       = 0
    END         = 1
    CLEAR       = 2

    LEN = 3
    NAME:dict = None

    @classmethod
    def name(cls):
        kv_dict = cls.__dict__
        name_dict = dict()
        for k, v in kv_dict.items():
            if isinstance(v, int):
                name_dict[v] = k

        cls.NAME = name_dict

Status.name()

                

pos_list = list()
curve_list = list()
curve_factor = 1.0
cur_pos = None
graph_status = Status.PAINT

img = cv2.imread(img_name)
if img is None:
    print("ERR: path is invalid!")
    print(os.listdir(img_dir))
    exit(-1)

def besel_curve_inside(in_pos_list):    

    factor = 0.2
    n_loop = 10

    tmp_list = np.copy(in_pos_list)
    for n in range(n_loop):

        new_list = list()
        for ii in range(len(tmp_list)):
            jj = (ii+1) % len(tmp_list)
            d = tmp_list[jj] - tmp_list[ii]
            d = d * factor
            new_list.append(tmp_list[ii] + d)
            new_list.append(tmp_list[jj] - d)
        
        tmp_list = np.array(new_list)

    return tmp_list

def besel_curve_p4(in_pos_list, t):    
    assert len(in_pos_list) == 4
    tmp_list = np.array(in_pos_list)
    t = t.reshape(-1, 1)
    sp = None

    p0 = np.power(1-t, 3) * tmp_list[0].reshape(1, -1)
    p1 = 3 * np.power(1-t, 2) * np.power(t, 1) * tmp_list[1].reshape(1, -1)
    p2 = 3 * np.power(1-t, 1) * np.power(t, 2) * tmp_list[2].reshape(1, -1)
    p3 = 1 * np.power(1-t, 0) * np.power(t, 3) * tmp_list[3].reshape(1, -1)
    return p0 + p1 + p2 + p3


def besel_curve_outside(in_pos_list):    

    factor = 0.2
    n = 100

    tmp_list = np.copy(in_pos_list)

    # - 生成扩展点
    new_list = list()
    for ii in range(len(tmp_list)):
        ij = (ii+1) % len(tmp_list)
        jj = (ii+2) % len(tmp_list)
        d = tmp_list[jj] - tmp_list[ii]
        d = d * factor
        new_list.append(tmp_list[ij] - d)
        new_list.append(tmp_list[ij])
        new_list.append(tmp_list[ij] + d)

    t = np.linspace(0, 1.0, n).reshape(-1, 1)
    new_list = np.array(new_list)

    result_list = list()
    for i in range(3):
        local_list = list()
        # - 点分组
        for j in range(4):
            local_list.append(new_list[(i*3 + 1 + j) % len(new_list)])
        
        local_list = np.stack(local_list)
        print(f' - local_list {local_list.shape}')

        result_list.append(besel_curve_p4(local_list, t))
        
    return np.concatenate(result_list, axis=0)



        


def reset():
    pos_list.clear()
    curve_list.clear()

def mouse_func(event, x, y, flags, params):
    global pos_list
    global graph_status
    global cur_pos
    global curve_factor
    cur_pos = [x, y]

    def modify_status():
        global graph_status
        old_status = graph_status
        graph_status += 1
        graph_status = graph_status % Status.LEN
        print(f'INFO: status {Status.NAME[old_status]} -> {Status.NAME[graph_status]}')

    if event == cv2.EVENT_LBUTTONDOWN:
        if graph_status == Status.PAINT:
            pos_list.append([x, y])
            print(f'INFO: Record pixel x {x}, y {y}')
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(pos_list) > 0 and graph_status == Status.PAINT:
            pos = pos_list.pop()
            print(f'INFO: delete pixel x {pos[0]}, y {pos[1]}')
        else:
            print("INFO: no pixel to delete!")
    elif event == cv2.EVENT_MBUTTONDOWN:
        modify_status()

        if graph_status == Status.CLEAR:
            reset()
            modify_status()

    elif event == cv2.EVENT_MOUSEWHEEL:
        # print(f' mouse move x {x}, y {y}')
        if graph_status == Status.END:
            curve_factor += 0.1 * y
            print(f'INFO: curve_factor {curve_factor:.3f}')

        

# cv2.imshow(win_name, paint_img)

def modify_line(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    # is_vertical
    # is_heritical
    

def Paint(image):
    
    # - 绘制点
    for pos in pos_list:
        cv2.circle(image, pos, 3, [0, 0, 255], thickness=-1)
        # cv2.circle(image, pos, 3, [255, 0, 0], )


    ex_pos_list = deepcopy(pos_list)
    if graph_status == Status.PAINT and len(pos_list) > 0:
        ex_pos_list.append(cur_pos)

    for ii in range(len(ex_pos_list)):
        jj = (ii + 1) % len(ex_pos_list)
        pt0 = np.array(ex_pos_list[ii])
        pt1 = np.array(ex_pos_list[jj])
        cv2.line(image, pt0.tolist(), pt1.tolist(), [255, 255, 0], 1)
        mid = (pt0 + pt1) * 0.5
        cv2.putText(image, 
                        # text='{:.3f}'.format(np.linalg.norm(pt1-pt0)), 
                        text=f'{np.linalg.norm(pt1-pt0):.3f}', 
                        org=mid.astype(int).tolist(), 
                        fontFace=cv2.FONT_HERSHEY_PLAIN, 
                        fontScale=1, 
                        color=[0])
        
    if graph_status == Status.END and len(pos_list) > 2:
        # curve_poses = besel_curve_inside(pos_list)
        curve_poses = besel_curve_outside(pos_list)
        gmt = geometry.Polygon(curve_poses).buffer(curve_factor)
        curve_poses = np.array(gmt.exterior.coords)


        curve_poses = np.round(curve_poses).astype(int)
        cv2.polylines(image, [curve_poses], True, [100, 100, 0])

    return image
    

    # if len(pos_list) == 1:
    #     cv2.KeyPoint(pos_list[0][0], pos_list[0][1], 0.2)
    # elif len(pos_list) > 1:
    #     pass

while True:
    paint_img = np.copy(img)
    Paint(paint_img)
    cv2.imshow(win_name, paint_img)
    cv2.setMouseCallback(win_name, mouse_func, ('a', 'b', 12, 34))

    if cv2.waitKey(300) == ord('q'):
        break