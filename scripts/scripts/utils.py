import os
import cv2
import math
import collections
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

def extract_data(path):
    gt_file = img_file = []
    list_files = os.listdir(path)
    for file in list_files:
        if file.endswith(".gt"):
            gt_file.append(file)
        if file.endswith(".JPG"):
            img_file.append(file)
    return img_file, gt_file

def extract2df(path):
    img_file, gt_file = extract_data(path)
    assert len(img_file) == len(gt_file)

    datas = collections.defaultdict(list)
    for gt in gt_file:
        name = gt.split(".")[0]
        with open(os.path.join(path, f'{name}.gt'), 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.split(" ")
            datas['image'].append(name)
            datas['index'].append(int(line[0]))
            datas['difficult'].append(int(line[1]))
            datas['x'].append(int(line[2]))
            datas['y'].append(int(line[3]))
            datas['w'].append(int(line[4]))
            datas['h'].append(int(line[5]))
            datas['theta'].append(float(line[6].split('\n')[0]))
            datas['degree'].append(np.round(np.degrees(np.arccos(float(line[6].split('\n')[0])))))

            cX = int(line[4]) // 2 + int(line[2])
            cY = int(line[5]) // 2 + int(line[3])

            datas['cx'].append(cX)
            datas['cy'].append(cY)
    
    return pd.DataFrame(datas)

def get_vertices(corner):
    vertice = []
    for cor in corner:
        for c in cor:
            vertice.append(c)
    return np.array(vertice)

def get_corners(box):
    x1, y1 = box[0], box[1]

    x2 = box[0] + box[2]
    y2 = y1

    x3 = x2
    y3 = box[1] + box[3]

    x4 = x1
    y4 = y3

    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

def cal_points(x, y, cx, cy, degree):
    degree = 90 - degree
    degree = np.radians(degree)
    x1 = np.round((np.cos(degree) * (x - cx)) - (np.sin(degree) * (y - cy)) + cx)
    y1 = np.round((np.sin(degree) * (x - cx)) + (np.cos(degree) * (y - cy)) + cy)
    return [int(x1), int(y1)]

def move_points(corners, cx, cy, degree):
    new_corners = []
    for coord in corners:
        new_corners.append(cal_points(coord[0], coord[1], cx, cy, degree))
    return new_corners

def get_mask(mask, points):
    cv2.fillPoly(mask, [points], color=(255, 255, 255))
    return mask

def adjust_box_sort(box):
    start = -1
    _box = list(np.array(box).reshape(-1, 2))
    min_x = min(box[0::2])
    min_y = min(box[1::2])
    _box.sort(key=lambda x:(x[0] - min_x)**2 + (x[1] - min_y)**2)
    start_point = list(_box[0])
    for i in range(0, 8, 2):
        x, y = box[i], box[i+1]
        if [x, y] == start_point:
            start = 1//2
            break

    new_box = []
    new_box.extend(box[start*2:])
    new_box.extend(box[:start*2])
    return new_box

def cal_distance(x1, y1, x2, y2):
    """Euclidean Distance"""
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def move_points_2(vertices, index1, index2, r, coef=0.3):
    index1 = index1 % 4
    index2 = index2 % 4

    x1_index = index1 * 2 + 0
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]
    r2 = r[index2]

    length_x = vertices[x1_index] - vertices[x2_index]
    length_y = vertices[y1_index] - vertices[y2_index]
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    
    return vertices

def shrink_poly(vertices, coef=0.3):

    x1, y1, x2, y2, x3, y3, x4, y4 = vertices

    r1 = min(cal_distance(x1, y1, x2, y2), cal_distance(x1, y1, x4, y4))
    r2 = min(cal_distance(x2, y2, x1, y1), cal_distance(x2, y2, x3, y3))
    r3 = min(cal_distance(x3, y3, x2, y2), cal_distance(x3, y3, x4, y4))
    r4 = min(cal_distance(x4, y4, x1, y1), cal_distance(x4, y4, x3, y3))

    r = [r1, r2, r3, r4]

    if cal_distance(x1, y1, x2, y2) + cal_distance(x3, y3, x4, y4) > cal_distance(x2, y2, x3, y3) + cal_distance(x1, y1, x4, y4):
        offset = 0
    else:
        offset = 1

    v = vertices.copy()
    v = move_points_2(v, 0 + offset, 1 + offset, r, coef)
    v = move_points_2(v, 2 + offset, 3 + offset, r, coef)
    v = move_points_2(v, 1 + offset, 2 + offset, r, coef)
    v = move_points_2(v, 3 + offset, 4 + offset, r, coef)

    return v

def get_rotate_mat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

def get_rotate_vertices(vertice, theta, anchor=None):
    v = vertice.reshape((4, 2)).T
    if anchor is None:
        anchor = v[:, :1]

    rotate_mat = get_rotate_mat(theta)
    res = np.dot(rotate_mat, v - anchor)
    return (res + anchor).T.reshape(-1)

def get_boundary(vertice):
    x1, y1, x2, y2, x3, y3, x4, y4 = vertice

    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)

    return x_min, x_max, y_min, y_max

def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    x = np.arange(length[0])
    y = np.arange(length[1])

    x, y = np.meshgrid(x, y)

    x_lin = x.reshape((1, x.size))
    y_lin = y.reshape((1, y.size))
    coord_mat = np.concatenate((x_lin, y_lin), 0)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + np.array([[anchor_x], [anchor_y]])
    rotated_x = rotated_coord[0, :].reshape(x.shape)
    rotated_y = rotated_coord[1, :].reshape(y.shape)

    return rotated_x, rotated_y

def get_score_geo(img, vertices, angles, scale, length):
    """
    Input :
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray (n,8)>
        labels      : 1 -> valid, 0 -> ignore
        scale       : feature map / image
        length      : image length

    Output :
        score_gt, geo_gt, ignored
    """
    H, W, _ = img.shape

    # W, H = img.size

    img_resize = cv2.resize(img, (int(W * scale), int(H * scale)), cv2.INTER_AREA)
    score_map = np.zeros((int(H * scale), int(W * scale), 1), np.float32)
    geo_map = np.zeros((int(H * scale), int(W * scale), 5), np.float32)


    # score_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    # geo_map = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)
    # ignored_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
    
    temp_masks = np.zeros(score_map.shape[:-1], np.float32)

    index_x = np.arange(0, length[0], int(1/scale))
    index_y = np.arange(0, length[1], int(1/scale))
    index_x, index_y = np.meshgrid(index_x, index_y)
    ignored_polys = []
    polys = []

    for i, vertice in enumerate(vertices):
        poly = np.around(scale * shrink_poly(vertice).reshape((4, 2))).astype(np.int32)
        polys.append(poly)
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        cv2.fillPoly(temp_mask, [poly], 1)
        cv2.fillPoly(temp_masks, [poly], 1)
        
        cv2.polylines(img_resize, [poly], True, (0, 0, 255), 5)

        theta = (angles[i] - 90) / 180 * math.pi

        rotate_mat = get_rotate_mat(theta)

        rotated_vertices = get_rotate_vertices(vertice, theta)
        
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        
        rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length)
        # print(vertice)
        d1 = rotated_y - y_min
        d1[d1<0] = 0
        d2 = y_max - rotated_y
        d2[d2<0] = 0
        d3 = rotated_x - x_min
        d3[d3<0] = 0
        d4 = x_max - rotated_x
        d4[d4<0] = 0

        geo_map[:, :, 0] += d1[index_y, index_x ] * temp_mask
        geo_map[:, :, 1] += d2[index_y, index_x ] * temp_mask
        geo_map[:, :, 2] += d3[index_y, index_x ] * temp_mask
        geo_map[:, :, 3] += d4[index_y, index_x ] * temp_mask
        geo_map[:, :, 4] += theta * temp_mask
    
    # cv2.fillPoly(ignored_map, ignored_polys, 1)
    cv2.fillPoly(score_map, polys, 1)

    # print(score_map.shape)
    # print(geo_map.shape)

    return score_map, geo_map

def Resize(df, w, h, w_r, h_r):
    x = int(np.round(df[3] * w_r / w))
    y = int(np.round(df[4] * h_r / h))
    w = int(np.round(df[5] * w_r / w))
    h = int(np.round(df[6] * h_r / h))
    
    return x, y, w, h

def main():
    path = '/home/kowlsss/Desktop/tutorial/torch/east/data/MSRA-TD500'
    train_path = os.path.join(path, 'train')

    train_df = extract2df(train_path)
    train_df.drop_duplicates(['image', 'index', 'difficult'], inplace=True)

    train_df['corners'] = train_df.apply(lambda df: get_corners([df.x, df.y, df.w, df.h]), axis=1)
    train_df['new_corners'] = train_df.apply(lambda df: move_points(df.corners, df.cx, df.cy, df.degree), axis=1)

    image_unique = train_df['image'].unique()
    ids = np.random.randint(len(image_unique))

    image_name = image_unique[ids]
    image = cv2.imread(os.path.join(train_path, f'{image_name}.JPG'))
    H, W, _ = image.shape
    # img = Image.fromarray(image)

    mask = np.zeros((H, W))
    df_select = train_df[train_df['image'] == image_name]
    plt.figure(figsize=(15, 12))
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 20))
    
    vertices = [] 
    angles = []
    scale = 0.25
    length = (W, H)

    for df in df_select.values:
        points = np.array(df[-1]).reshape((-1, 1, 2))

        cv2.polylines(image, [points], True, (0, 0, 255), 5)

        mask = get_mask(mask, points)
        src_pts = points.astype('float32')
        dst_pts = np.array([[0, df[6] - 1],
                        [0, 0],
                        [df[5] - 1, 0],
                        [df[5] - 1, df[6] - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (df[6], df[5]))
        
        angles.append(df[8])
        vertices.append(get_vertices(df[-1]))
   
    print(get_score_geo(image, vertices, angles, scale, length))


if __name__ == "__main__":
    main()
