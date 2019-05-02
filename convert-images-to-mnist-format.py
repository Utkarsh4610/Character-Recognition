from PIL import Image, ImageFilter
import os
import pandas as pd

def imageprepare(argv):
   # print(argv)
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels
    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
        # newImage.save("sample.png
    tv = list(newImage.getdata())  # get pixel values
    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return (tva)

import string
alpha = list(string.ascii_uppercase)
alpha = list(range(0,10))
for xx in alpha:

    data_storage = pd.DataFrame(columns=range(784))

    for f in os.listdir('F:/Machine Learning/jpg_to_mnist/test'):
        x = imageprepare('test/'+f)#file path here
        y = pd.DataFrame(x).T
        data_storage = data_storage.append(y)

    data_storage.insert(0, 'Character', 'Q' , allow_duplicates=False) # Need to change third argument as character..
    """Adding the dependent variable which is  
    character to be predicted in this case all are 0. So adding a column named Character with all values 0. 
    We need to run this whole algo for all the characters and at last need to append all. """
    #But there is a problem with index(All index is 0) need to fix it.
    data_storage.insert(0, 'Index', range(data_storage['Character'].size), allow_duplicates=False) 
    final_data_x= data_storage.set_index('Index',drop='True') #Name will change like for A final_data_A..etc
    final_data_x.to_csv('F:/Machine Learning/jpg_to_mnist/test.csv')


