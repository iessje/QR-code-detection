
from matplotlib import pyplot
from matplotlib.patches import Rectangle

import imageIO.png
import numpy as np
import math

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)
    
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# This method packs together three individual pixel arrays for r, g and b values into a single array that is fit for
# use in matplotlib's imshow method
def prepareRGBImageForImshowFromIndividualArrays(r,g,b,w,h):
    rgbImage = []
    for y in range(h):
        row = []
        for x in range(w):
            triple = []
            triple.append(r[y][x])
            triple.append(g[y][x])
            triple.append(b[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage
    

# This method takes a greyscale pixel array and writes it into a png file
def writeGreyscalePixelArraytoPNG(output_filename, pixel_array, image_width, image_height):
    # now write the pixel array as a greyscale png
    file = open(output_filename, 'wb')  # binary mode is important
    writer = imageIO.png.Writer(image_width, image_height, greyscale=True)
    writer.write(file, pixel_array)
    file.close()

def convertToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b):
    px_array_gs = []
    for i in range(image_height):
        row = []
        for j in range(image_width):
            #weighted covertion
            row.append(int(px_array_r[i][j]*0.299 + px_array_g[i][j]*0.587 + px_array_b[i][j]*0.114))
        px_array_gs.append(row)
    return px_array_gs

def normalize(image_width, image_height, pixel_array,):
    #print(len(pixel_array),len(pixel_array[0]))
    low = min([min(r) for r in pixel_array])
    high = max([max(r) for r in pixel_array])
    img = []
    if high == low:
        return img
    for i in range(image_height):
        row = []
        for j in range(image_width):
            row.append(int(round((pixel_array[i][j] - low) * (255/(high-low)))))
        img.append(row)
    return img

def convolve(a,b):
    #res =a[0][0]*b[2][2] + a[0][1]*b[2][1]+a[0][2]*b[2][0]+a[1][0]*b[1][2] + a[1][1]*b[1][1]+a[1][2]*b[1][0]+a[2][0]*b[0][2] + a[2][1]*b[0][1]+a[2][2]*b[0][0]
    res = []
    for i in range(3):
        for j in range(3):
            res.append(a[i][j]*b[i][j])    
    return sum(res)
            
def sobelFilter(image_width, image_height, gs):
    xfilter = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    yfilter = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    sobel = [[0 for x in range(image_width)] for y in range(image_height)]

    for i in range(image_height-2):
        for j in range(image_width-2):
            x = convolve(xfilter,[gs[i][j:j+3],gs[i+1][j:j+3],gs[i+2][j:j+3]])
            y = convolve(yfilter,[gs[i][j:j+3],gs[i+1][j:j+3],gs[i+2][j:j+3]])
            sobel[i+1][j+1] = math.sqrt(x*x+y*y)
    #print(len(sobel),len(sobel[0]))
    return sobel

def meanFilter(image_width, image_height, p):
    window = 5
    c = int((window-1)/2) #border
    t = createInitializedGreyscalePixelArray(image_width+c*2, image_height+c*2)
    for i in range(c,image_height+c):
        for j in range(c,image_width+c):
            t[i][j] =  p[i-c][j-c]
            
    res = []
    #border ignore
    for i in range(c,image_height + c):
        row = []
        for j in range(c,image_width + c):
            total = 0
            for k in range(-c,c+1):
                total += sum(t[i+k][j-c:j+c+1])
            row.append(int(round(total/(window*window))))
        res.append(row)
    return res

def thresholdFilter(image_width, image_height, gs):
    t = 70 #threshold
    img = []
    for i in range(image_height):
        row = []
        for j in range(image_width):
            if gs[i][j] > t:
                row.append(255)
            else:
                row.append(0)
        img.append(row)
    return img

def createInitializedGreyscalePixelArray(image_width, image_height):
    p = []
    for i in range(image_height):
        row =[]
        for j in range(image_width):
            row.append(0)
        p.append(row)
    return p
    
def dilation(p, image_width, image_height):
    window = 3
    c = int((window-1)/2) #border
    t = createInitializedGreyscalePixelArray(image_width+2, image_height+2)
    #t = zero padded array
    for i in range(1,image_height+1):
        for j in range(1,image_width+1):
            t[i][j] =  p[i-1][j-1]
    a = []
    for i in range(c,image_height+c): #+c because array is padded
        row = []
        for j in range(c,image_width+c):
            total = 0
            for k in range(-c,c+1):
                for x in t[i+k][j-c:j+c+1]:
                    if not(x) == False:
                        total = 255
                        break;
            row.append(total)
        a.append(row)
    return a

def erosion(p, image_width, image_height):
    window = 3 
    c = int((window-1)/2) #border
    t = createInitializedGreyscalePixelArray(image_width+2, image_height+2)
    #t = zero padded array
    for i in range(1,image_height+1):
        for j in range(1,image_width+1):
            t[i][j] =  p[i-1][j-1]
    
    a = []
    for i in range(c,image_height+c): #+c because array is padded
        row = []
        for j in range(c,image_width+c):
            total = 255
            for k in range(-c,c+1):
                for x in t[i+k][j-c:j+c+1]:
                    if x == 0:
                        total = 0
                        break;
            row.append(total)
        a.append(row)
    return a

def ccl(p, image_width, image_height):
    
    current_label = 1
    visited = set()
    objects = {}
    cc = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if p[i][j] != 0 and (i,j) not in visited:
                visited.add((i,j))
                q = Queue()
                q.enqueue((i,j))
                objects[current_label] = 0
                while q.isEmpty() == False:
                    n = q.dequeue()
                    objects[current_label] += 1
                    x = n[0]#height
                    y = n[1]#width
                    cc[x][y] = current_label
                    if y-1>=0 and p[x][y-1] != 0 and (x,y-1) not in visited: #left
                        visited.add((x,y-1))
                        q.enqueue((x,y-1))
                    if y+1<=image_width-1 and p[x][y+1] != 0 and (x,y+1) not in visited: #right
                        visited.add((x,y+1))
                        q.enqueue((x,y+1))
                    if x-1>=0 and p[x-1][y] != 0 and (x-1,y) not in visited: #top
                        visited.add((x-1,y))
                        q.enqueue((x-1,y))
                    if x+1<=image_height-1 and p[x+1][y] != 0 and (x+1,y) not in visited: #bottom
                        visited.add((x+1,y))
                        q.enqueue((x+1,y))
                current_label += 1
    
    comp = max(objects, key = objects.get)

    for i in range(image_height):
        for j in range(image_width):
            if cc[i][j] == comp:
                cc[i][j] = 255
            else:
                cc[i][j] = 0
    return cc
                    
def find_box(p, image_width, image_height):
    x = []
    for i in range(image_height):
        for j in range(image_width):
            if p[i][j] != 0:
                x.append(j)
                break;
    y = []
    for i in range(image_width):
        for j in range(image_height):
            if p[j][i] != 0:
                y.append(j)
                break;
    w = []
    for i in range(image_height-1,-1,-1):
        for j in range(image_width-1,-1,-1):
            if p[i][j] != 0:
                w.append(j)
                break;

    h = []
    for i in range(image_width-1,-1,-1):
        for j in range(image_height-1,-1,-1):
            if p[j][i] != 0:
                h.append(j)
                break;
            
    return min(x),min(y),max(w),max(h)

def main():
    filename = "./images/covid19QRCode/poster1small.png"

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(filename)
    
    #CONVERT RGB ARRAYS TO GREYSCALE ARRAY
    print("CONVERT RGB ARRAYS TO GREYSCALE ARRAY")
    px_array_gs = convertToGreyscale(image_width, image_height, px_array_r, px_array_g, px_array_b)
    px_array_gs = normalize(image_width, image_height,px_array_gs)

    #CREATE GREYSCALE PNG
    #print("CREATE GREYSCALE PNG")
    #writeGreyscalePixelArraytoPNG("1greyimg.png", px_array_gs, image_width, image_height)
    
    #PERFORM SOBEL FILTER ON GREYSCALE ARRAY
    print("PERFORM SOBEL FILTER ON GREYSCALE ARRAY")
    px_array_sobel = sobelFilter(image_width, image_height, px_array_gs)
    px_array_sobel = normalize(image_width, image_height, px_array_sobel)

    #CREATE GREYSCALE PNG
    #print("CREATE GREYSCALE PNG")
    #writeGreyscalePixelArraytoPNG("2sobelimg.png", px_array_sobel, image_width, image_height)

    #PERFORM MEAN FILTER THRICE ON SOBEL FILTER
    print("PERFORM MEAN FILTER THRICE ON SOBEL FILTER")
    px_array_mean = meanFilter(image_width, image_height, px_array_sobel)
    px_array_mean = normalize(image_width, image_height, px_array_mean)
    px_array_mean = meanFilter(image_width, image_height, px_array_mean)
    px_array_mean = normalize(image_width, image_height, px_array_mean)
    px_array_mean = meanFilter(image_width, image_height, px_array_mean)
    px_array_mean = normalize(image_width, image_height, px_array_mean)

    #CREATE GREYSCALE PNG
    #print("CREATE GREYSCALE PNG")
    #writeGreyscalePixelArraytoPNG("3meanimg.png", px_array_mean, image_width, image_height)

    #PERFORM THRESHOLD FILTER ON MEAN FILTER
    print("PERFORM THRESHOLD FILTER ON MEAN FILTER")
    px_array_thres = thresholdFilter(image_width, image_height, px_array_mean)

    #CREATE GREYSCALE PNG
    #print("CREATE GREYSCALE PNG")
    #writeGreyscalePixelArraytoPNG("4thresimg.png", px_array_thres, image_width, image_height)

    #PERFORM  DILATION FILTER ON THRESHOLD FILTER
    print("PERFORM DILATION FILTER ON THRESHOLD FILTER")
    px_array_dial = dilation(px_array_thres, image_width, image_height)
    px_array_dial = dilation(px_array_dial, image_width, image_height)
    
    #CREATE GREYSCALE PNG
    #print("CREATE GREYSCALE PNG")
    #writeGreyscalePixelArraytoPNG("5dialimg.png", px_array_dial, image_width, image_height)
    
    #PERFORM  EROSION FILTER ON DILATION  FILTER
    print("PERFORM EROSION FILTER ON DILATION  FILTER")
    px_array_eros = erosion(px_array_dial, image_width, image_height)
    px_array_eros = erosion(px_array_eros, image_width, image_height)
    
    #CREATE GREYSCALE PNG
    #print("CREATE GREYSCALE PNG")
    #writeGreyscalePixelArraytoPNG("6erosimg.png", px_array_eros, image_width, image_height)

    #PERFORM  CONNECTED COMPONENT LABELING
    print("PERFORM CONNECTED COMPONENT LABELING")
    px_array_ccl = ccl(px_array_eros, image_width, image_height)

    #CREATE GREYSCALE PNG
    #print("CREATE GREYSCALE PNG")
    #writeGreyscalePixelArraytoPNG("7cclimg.png", px_array_ccl, image_width, image_height)

    #FIND BOUNDING BOX
    print("FIND BOUNDING BOX")
    (x,y,w,h) = find_box(px_array_ccl, image_width, image_height)

    pyplot.imshow(prepareRGBImageForImshowFromIndividualArrays(px_array_r, px_array_g, px_array_b, image_width, image_height))

    # get access to the current pyplot figure
    axes = pyplot.gca()
    # create a 70x50 rectangle that starts at location 10,30, with a line width of 3
    #rect = Rectangle( (10, 30), 70, 50, linewidth=3, edgecolor='g', facecolor='none' )


    rect = Rectangle( (x, y), w-x, h-y, linewidth=3, edgecolor='g', facecolor='none' )
    # paint the rectangle over the current plot
    axes.add_patch(rect)

    # plot the current figure
    pyplot.show()



if __name__ == "__main__":
    main()
