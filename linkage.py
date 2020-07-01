from PIL import Image

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from  scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
def all_linkage(y):
    try:
        single_z=linkage(y,'single')
    except ValueError:
        print("ha")
    complete_z=linkage(y,'complete')
    average_z = linkage(y,'average')
    centroid_z=linkage(y,'centroid')
    return {'single':single_z,'complete':complete_z,'average':average_z,'centroid':centroid_z}
def all_pdist(x):
    euclidean_y=pdist(x,'euclidean')
    seuclidean_y = pdist(x, 'seuclidean')
    array_sum = np.sum(seuclidean_y)
    array_has_nan = np.isnan(array_sum)
    if array_has_nan:
        seuclidean_y=euclidean_y
    return {'euclidean':euclidean_y,'seuclidean':seuclidean_y}

def all_dendogram(x,img):
    dict={}
    t=1
    # for i in range(10000, len(x), 10000):
    for key,value in all_pdist(x).items():
        for key1,value1 in all_linkage(value).items():
                try:
                    dict["linkage_method_"+key1+"_pdist_type_"+key+"_t_"+str(t)].extend(fcluster(value1,criterion='distance',t=t))
                except KeyError:
                    dict["linkage_method_" + key1 + "_pdist_type_" + key + "_t_" + str(t)]=[]
                    dict["linkage_method_" + key1 + "_pdist_type_" + key + "_t_" + str(t)].extend(fcluster(value1, criterion='distance',t=t))
    # if i<len(x):
    #     for key,value in all_pdist(x[i:]).items():
    #         for key1,value1 in all_linkage(value).items():
    #                 dict["linkage_method_"+key1+"pdist_type_"+key+"_t_0.1"].extend(fcluster(value1,criterion='distance',t=0.1))

    for key,array in dict.items():
        identifiers=get_label_dict(x,array)

        imag_filtering(array,img,identifiers,key)




def read_file(file_name: str) :
    img = Image.open(file_name, 'r')
    pix_val = list(img.getdata())
    return img,np.asarray(pix_val, dtype=np.int)

def imag_filtering(y, img: Image, identifiers,save_name):
    new_image = []
    for x in range(img.size[1]):
        new_image_row = []
        for i in range(img.size[0]):
            new_image_row.append(identifiers[y[x * (img.size[0]) + i]])
        new_image.append(new_image_row)
    new_image = np.asarray(new_image, dtype=np.uint8)
    new_image = Image.fromarray(new_image, 'RGB')
    new_image.save(save_name+'.jpg')
    new_image.show()

def get_label_dict(X_train, y_train) -> dict:
    result={}
    for label in np.unique(y_train):
        result[label]=[]
        for i in range(len(X_train)-1):
            if label==y_train[i]:
                result[label].append(X_train[i])
    for key,value in result.items():
        result[key]=np.mean(value,axis=0)



    return result


img, image_data = read_file('mcdonalds-Fries-Small.jpg')
image_data=np.array(image_data)
all_dendogram(image_data,img)