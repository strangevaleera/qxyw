#coding=utf-8
import h5py
import numpy as np
import caffe

def img_transform(img_name, input_shape):

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = cv2.imread(img_name)
    image = cv2.resize(image, input_shape, interpolation=cv2.INTER_CUBIC)
    image = image[..., ::-1]  # BGR to RGB
    image = image.astype(np.float32) / 255.  # uint8 to float32
    # Normalization
    image = (image - mean[None, None, :]) / std[None, None, :]
    image = image.transpose([2, 0, 1])

    return image

#1.导入数据  数据集没确定
filename = 'testdata.h5'
f = h5py.File(filename, 'r')
n1 = f.get('data')
n1 = np.array(n1)
print n1[0]
n2=f.get( 'label_1d')
n2 = np.array(n2)
f.close()

#2.导入模型与网络
deploy='shufflenet_v2_x0.5.prototxt'    #deploy文件
caffe_model= 'shufflenet_v2_x0.5.caffemodel'   #训练好的 caffemodel
net = caffe.Net(deploy,caffe_model,caffe.TEST)


count=0   #统计预测正确的数量
n=n1.shape[0]    #n：样本的数量
for i in range(n):
    input_shape = net.blobs['blob1'].data.shape[2:]
    image = img_transform(n1[i], input_shape)
    net.blobs['blob1'].data[...] = image
    out = net.forward()
    output = out['outputs']
    result= np.where(output==np.max(output))
    predi=result[1][0]
    #判断predi与label是否相等，并统计
    label = n2[i, 0]
    if predi==(label):
     count=count+1
    kk=[predi,label]
    print kk
print count