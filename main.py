
from cv2 import cv2
import numpy as np
from numpy import linalg as LA



def extractingColorChannel(channel, cell_size, w, h):
    channel_his = []
    for cx in range(w // cell_size):
        for cy in range(h // cell_size):
            channel_xy = channel[cy * cell_size:cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]
            channel_his_xy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(cell_size):
                for j in range(cell_size):
                    if (channel_xy[i][j] >= 0) and (channel_xy[i][j] < 25):
                        channel_his_xy[0] += 1
                    elif (channel_xy[i][j] >= 25) and (channel_xy[i][j] < 50):
                        channel_his_xy[1] += 1
                    elif (channel_xy[i][j] >= 50) and (channel_xy[i][j] < 50):
                        channel_his_xy[2] += 1
                    elif (channel_xy[i][j] >= 75) and (channel_xy[i][j] < 50):
                        channel_his_xy[3] += 1
                    elif (channel_xy[i][j] >= 100) and (channel_xy[i][j] < 50):
                        channel_his_xy[4] += 1
                    elif (channel_xy[i][j] >= 125) and (channel_xy[i][j] < 50):
                        channel_his_xy[5] += 1
                    elif (channel_xy[i][j] >= 150) and (channel_xy[i][j] < 50):
                        channel_his_xy[6] += 1
                    elif (channel_xy[i][j] >= 175) and (channel_xy[i][j] < 50):
                        channel_his_xy[7] += 1
                    elif (channel_xy[i][j] >= 200) and (channel_xy[i][j] < 50):
                        channel_his_xy[8] += 1
                    elif (channel_xy[i][j] >= 225) and (channel_xy[i][j] <= 255):
                        channel_his_xy[9] += 1
                pass
            pass
            channel_his = channel_his + channel_his_xy
            # print(channel_his)
        pass
    pass
    return channel_his


def extractingColor(img_path, cell_size=8):
    img = cv2.imread(img_path)
    img = cv2.resize(src=img, dsize=(64, 128))
    h, w, _ = img.shape
    b, g, r = cv2.split(img)

    hb = extractingColorChannel(b, cell_size, w, h)
    hg = extractingColorChannel(g, cell_size, w, h)
    hr = extractingColorChannel(r, cell_size, w, h)

    histogramColor = hb + hg + hr
    # print(len(histogramColor))
    histogramColor = np.array(histogramColor)
    # print(histogramColor)
    histogramColor = histogramColor / LA.norm(histogramColor, 2)
    np.seterr(divide='ignore', invalid='ignore')

    return histogramColor







def extractingShape(img_path, cell_size=8, block_size=2, bins=9):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(src=img, dsize=(64, 128))
    h, w = img.shape  # 128, 64

    # gradient
    xkernel = np.array([[-1, 0, 1]])
    ykernel = np.array([[-1], [0], [1]])
    dx = cv2.filter2D(img, cv2.CV_32F, xkernel)
    dy = cv2.filter2D(img, cv2.CV_32F, ykernel)

    # histogram
    magnitude = np.sqrt(np.square(dx) + np.square(dy))
    orientation = np.arctan(np.divide(dy, dx + 0.00001))  # radian
    orientation = np.degrees(orientation)  # -90 -> 90
    orientation += 90  # 0 -> 180

    num_cell_x = w // cell_size  # 8
    num_cell_y = h // cell_size  # 16

    hist_tensor = np.zeros([num_cell_y, num_cell_x, bins])  # 16 x 8 x 9
    for cx in range(num_cell_x):
        for cy in range(num_cell_y):
            ori = orientation[cy * cell_size:cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]
            mag = magnitude[cy * cell_size:cy * cell_size + cell_size, cx * cell_size:cx * cell_size + cell_size]
            hist, _ = np.histogram(ori, bins=bins, range=(0, 180), weights=mag)  # 1-D vector, 9 elements
            hist_tensor[cy, cx, :] = hist
        pass
    pass

    # normalization
    redundant_cell = block_size - 1
    feature_tensor = np.zeros(
        [num_cell_y - redundant_cell, num_cell_x - redundant_cell, block_size * block_size * bins])
    for bx in range(num_cell_x - redundant_cell):  # 7
        for by in range(num_cell_y - redundant_cell):  # 15
            by_from = by
            by_to = by + block_size
            bx_from = bx
            bx_to = bx + block_size
            v = hist_tensor[by_from:by_to, bx_from:bx_to, :].flatten()  # to 1-D array (vector)
            feature_tensor[by, bx, :] = v / LA.norm(v, 2)
            np.seterr(divide='ignore', invalid='ignore')
            # avoid NaN:
            if np.isnan(feature_tensor[by, bx, :]).any():  # avoid NaN (zero division)
                feature_tensor[by, bx, :] = v

    return feature_tensor.flatten()  # 3780 features
def distance(x1,x2):
    x=x1-x2
    x=x**2
    x=np.sum(x)
    return np.sqrt(x)

def predict(x):
    distances=[]
    for feature in X:
        distances.append(distance(x,feature))
    distances=np.array(distances)
    index=np.argmin(distances)
    return labels[y[index]]






def test(file_name):

    img = cv2.imread('test3/' + file_name)
    color = extractingColor('test3/' + file_name)
    shape = extractingShape('test3/' + file_name)
    v = np.concatenate((color, shape), axis=0)

    #predict
    result=predict(v)


    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, result, (32, 64), font, 2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('test3', img)
    cv2.waitKey(0)



if __name__ == '__main__':
   data=np.load('data.npy')
   labels={0:'doraemon',1:'jerry',2:'nobita',3:'tom'}
   X=data[:,:-1]
   y=data[:,-1]


   while(True):
       filename=input()
       if(filename=='0'):
           break
       test(filename)

