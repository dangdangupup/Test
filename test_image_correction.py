"""
    测试：图片修正
"""

# =====================================
import cv2
import numpy as np
import matplotlib.pyplot as plt


CV_WIN_NAME = 'Get'
CV_WIN_NAME_ORG = 'Org'
CV_WIN_NAME_MD = 'Modify'

data_list = list()

def event_func(event, x, y, flags, param):
    global data_list
    global keypoint_path

    if event == cv2.EVENT_FLAG_LBUTTON:
        print(f"OP: left button click  {event, x, y}")
        data_list.append([x, y])
    elif event == cv2.EVENT_FLAG_RBUTTON:
        print(f"OP: save key points")
        np.savetxt(keypoint_path, np.array(data_list))


def draw_keypoint(img, key_points, color = (255)):
    key_points = np.array(key_points).astype(np.uint16)
    for kp in key_points:
        img = cv2.circle(img, kp.astype(np.uint16), 1, color)
    return img

def get_ref_point(img_path):
    org_data = cv2.imread(img_path)
    org_data = cv2.cvtColor(org_data, cv2.COLOR_BGR2GRAY)

    if org_data is None:
        print(f"Error: path: {img_path}")
        exit(-1)
    while True:
        out_data = np.copy(org_data)

        # - 自动获取角点，不够精准
        # kps = cv2.goodFeaturesToTrack(out_data, 10000, qualityLevel=0.01, minDistance=10, blockSize=3))
        # d = cv2.cornerSubPix(out_data, np.copy(kps), (5,5), (-1, -1), None)
        # kps = kps.reshape(-1, 2)

        out_data = draw_keypoint(out_data, data_list)

        cv2.imshow(CV_WIN_NAME, out_data)
        cv2.setMouseCallback(CV_WIN_NAME, event_func)
        key = cv2.waitKey(100)
        if key == ord('q'):
            exit(-1)
        elif key == ord('z'):
            print("Back...")
            if len(data_list) > 0: 
                data_list.pop()
        

def normalize(data_list, is_denoise = False):

    # # - 续尾巴
    # N = 30
    # diff = data_list[-1] - data_list[-2]
    # data_list = list(data_list)
    # for i in range(N):
    #     data_list.append(data_list[-1] + np.exp(-i)*diff)
    # data_list = np.array(data_list)

    # - TODO remove noise
    if is_denoise:
        tmp = data_list - data_list.mean(axis=0)
        res = np.linalg.eig(tmp.T @ tmp)
    
    # - 归 0
    ref_0 = data_list[0] # (x, y)
    data = np.linalg.norm(data_list - ref_0, axis=-1)

    # - 归 1, 参考（第1个值，或者中间值）
    ref_1 = (data[1] - data[0]) * 0.5 # scale
    data /= ref_1
    return data, [ref_0, ref_1]
    

def modify_image_inv(in_img, fit_f, refs):
    # out_img = np.copy(in_img)
    out_img = np.zeros(in_img.shape).astype(np.uint8)

    ref_0 = np.array(refs[0])
    ref_0[::-1] = ref_0
    ref_1 = refs[1]
    print(f"INFO: ref_0 {ref_0}, ref_1 {ref_1}")

    h, w = in_img.shape

    hws_org = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
    hws_m = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
    hws_m = np.dstack(hws_m)

    lens_m = np.linalg.norm(hws_m - ref_0, axis=-1, keepdims=True)
    lens_m += 1e-6
    cos_sin_x = (hws_m - ref_0) / lens_m # 角度，用于还原 x,y

    lens_m /= ref_1
    lens = fit_f(lens_m)
    lens *= ref_1

    hws = lens * cos_sin_x + ref_0
    print(f"INFO: min max {hws.min(), hws.max()}")

    hws = np.clip(hws, 0, 1023)
    hws = hws.astype(int) # TODO: 使用插值，提高精度
    out_img[hws_org[0].astype(int), hws_org[1].astype(int)] = in_img[hws[:, :, 0], hws[:, :, 1]]

    return out_img

# - NOTE: has problem with xxx
def modify_image(in_img, fit_f, refs):
    # out_img = np.copy(in_img)
    out_img = np.zeros(in_img.shape).astype(np.uint8)

    ref_0 = np.array(refs[0])
    ref_0[::-1] = ref_0
    ref_1 = refs[1]
    print(f"INFO: ref_0 {ref_0}, ref_1 {ref_1}")

    h, w = in_img.shape

    hws_org = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
    hws = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
    hws = np.dstack(hws)

    lens = np.linalg.norm(hws - ref_0, axis=-1, keepdims=True)
    lens += 1e-6
    cos_sin_x = (hws - ref_0) / lens # 角度，用于还原 x,y

    lens /= ref_1
    lens_m = fit_f(lens)
    lens_m *= ref_1

    hws_m = lens_m * cos_sin_x + ref_0
    print(f"INFO: min max {hws_m.min(), hws_m.max()}")

    hws_m = np.clip(hws_m, 0, 1023)
    hws_m = hws_m.astype(int) # TODO: 使用插值，提高精度
    # out_img[hws_org[0].astype(int), hws_org[1].astype(int)] = in_img[hws[:, :, 0], hws[:, :, 1]]
    out_img[hws_m[:, :, 0], hws_m[:, :, 1]] = in_img[hws_org[0].astype(int), hws_org[1].astype(int)]

    return out_img
    
def main(img_path, keypoint_path):

    
    # get_ref_point(path)

    data_list = np.loadtxt(keypoint_path)

    # - preprocess data
    vs, refs = normalize(data_list)
    xs = np.arange(len(vs))

    # - fit data
    fit_p = np.polyfit(xs, vs, 4)
    test_fit_p = np.polyfit(vs, xs, 4)
    print(f" - test_fit_p {['%.6f'%p for p in test_fit_p]}")
    fit_f = np.poly1d(fit_p)
    fit_vs = fit_f(np.arange(50))

    # - display: curve
    plt.plot(xs, vs, label='x-y')
    plt.plot(vs, xs, label='y-x')
    plt.plot(np.arange(50), fit_vs, label='fit-y-x')
    plt.legend()
    plt.ion()
    plt.show()


    org_img = cv2.imread(img_path, 0)
    out_img = modify_image_inv(org_img, fit_f, refs)

    # - NOTE: org keypoints
    org_kps = np.stack([vs, np.zeros(len(vs))]).T
    org_kps = org_kps * refs[1] + refs[0]
    org_img = draw_keypoint(org_img, key_points=org_kps)

    cv2.imshow(CV_WIN_NAME_ORG, org_img)
    cv2.imshow(CV_WIN_NAME_MD, out_img)
    cv2.waitKey(-1)


if __name__ == '__main__':
    target = 'bw'
    img_path = f'/Users/ed/Data/Image/{target}_fisheye.png'
    keypoint_path = f'keypoint_{target}.txt'
    # get_ref_point(img_path=img_path)
    main(img_path, keypoint_path)


