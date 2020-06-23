import os
from sys import platform
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def cosine_similarity(image1, image2):
    image1Vector = np.array(image1.getdata()).astype('float32').ravel()
    image2Vector = np.array(image2.getdata()).astype('float32').ravel()
    return np.dot(image1Vector, image2Vector) / ( np.linalg.norm(image1Vector) * np.linalg.norm(image2Vector) )
    

def readImages(folderpath):
    images = {}
    sign = "\\" if platform == "win32" else "//"
    for filename in os.listdir(folderpath):
        images.update( { filename : Image.open(folderpath + sign + filename) } )
    return images

def getSimilarityList(inputImageName, images, n):
    try:
        inputImageData = images[inputImageName]
    except:
        raise KeyError("Specified input image name could not found in images")
    images_w_sim = {} # name : similarity
    size = len(images)
    if n > size:
        raise IndexError("N cannot be large than images size")
    for filename in images.keys():
        if(inputImageName != filename):
            images_w_sim.update({filename: cosine_similarity(inputImageData, images[filename])})
    return [[filename, images_w_sim[filename]] for filename in sorted(images_w_sim, key=images_w_sim.get, reverse=True)[:n]]


def plotSimilarity(inputImageName, similarImages, images):
    fig=plt.figure(figsize=(8, 8))
    
    total = len(similarImages) + 1
    columns = 5
    rows = int(total / columns) + (total % columns > 0)
     
    fig.add_subplot(rows, columns, 1)
    plt.title("Input")
    plt.imshow(images[inputImageName])
    for i in range(total - 1):
        fig.add_subplot(rows, columns, i+2)
        filename, similarity = similarImages[i][0], similarImages[i][1]
        plt.title(similarity.round(7))
        plt.imshow(images[filename])
    plt.show()



if __name__ == '__main__':
    folderpath = "Car_Data"
    n = 3

    images = readImages(folderpath) # { filename1:colorfulimage1, filename2:colorfulimage2 }
    
    inputImageName = '4228.png'
    similarImages = getSimilarityList(inputImageName, images, n) # [ [filename1, similarity1] , [filename2, similarity2] , ... ]
    plotSimilarity(inputImageName, similarImages, images)

    inputImageName = '3861.png'
    similarImages = getSimilarityList(inputImageName, images, n) # [ [filename1, similarity1] , [filename2, similarity2] , ... ]
    plotSimilarity(inputImageName, similarImages, images)



