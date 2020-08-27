import cv2
import numpy as np
import operator
import matplotlib.pyplot as plt


def plot_all_boxes(images,L,W):
    fig,axes=plt.subplots(L,W,figsize=(28,28))
    axes=axes.ravel()
   
    for i in range(0,L*W):
        axes[i].imshow(images[i],cmap='gray')
        axes[i].axis('off')
        plt.subplots_adjust(wspace=0)
    plt.show()
def show_image(x):
    # commented the function as wrote new one below
    # cv2.imshow(x)
    return x
def show_digits(digits, colour=255,imageshow=False):

    rows = []
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
    img = show_image(np.concatenate(rows))
    return img

    return rows

def readimg(path):
    image = cv2.imread(path)
    simage = cv2.resize(image, (450, 450),interpolation = cv2.INTER_NEAREST) 
    return image

def process_image(img,skip_dilate=False):
    """Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""

    # Gaussian blur with a kernal size (height, width) of 9.
    # Note that kernal sizes must be positive and odd and the kernel must be square.
    proc = cv2.GaussianBlur(img.copy(), (9, 9), 0)

    # Adaptive threshold using 11 nearest neighbour pixels
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert colours, so gridlines have non-zero pixel values.
    # Necessary to dilate the image, otherwise will look like erosion instead.
    proc = cv2.bitwise_not(proc, proc)

    if not skip_dilate:
        # Dilate the image to increase the size of the grid lines.
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
        proc = cv2.dilate(proc, kernel)

    return proc


def showimg(img,title=''):
    cv2.imshow(title,img)
    cv2.waitKey(0)


def get_points(thres):
    contours, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]



def plotpoints(image,r,*points,square=False):
    points = np.squeeze(points)
    if square:
        for point in points:
            for pt in point:
                image = cv2.circle(image, tuple(pt.astype('int')), radius=r, color=(0, 0, 255), thickness=-1)
    else:
        for point in points:
            image = cv2.circle(image, tuple(point.astype('int')), radius=r, color=(255, 128, 120), thickness=-1)
    return image
    


def distance_between(p1, p2):

    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def image_cropper(img,crop_rect):

    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    
    longestside = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])
    dst = np.array([[0, 0], [longestside-1 , 0], [longestside-1 , longestside -1], [0, longestside -1]], dtype='float32')

    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (int(longestside), int(longestside)))



def reduceintopieces(img):
    """Infers 81 cell grid from a square image."""
    squares = []
    side = img.shape[:1]
    side = side[0] / 9
    p3s = []
    # Note that we swap j and i here so the rectangles are stored in the list reading left-right instead of top-down.
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)  # Top left corner of a bounding box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
            squares.append((p1, p2))
            p3 = (i * side + side /1.4 -side/3, j * side +side /1.4)
            p3s.append(p3)
    return squares,p3s

 

def generate_digits(img,rect):
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]



def scaling(img,size,margin,bg = 0):
    h,w = img.shape[:2]

    def centre(length):
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    if h > w:
        tp = int(margin/2)
        bp = int(margin/2)
        r = (size - margin) / h
        h,w = int(r*h),int(r*w) 
        lp,rp = centre(w)

    else:
        lp = int(margin/2)
        rp = lp 
        r = (size - margin)/w
        h,w = int(r*h),int(w*h)
        tp,bp = centre(h)

    img = cv2.resize(img,(w,h))
    img = cv2.copyMakeBorder(img, tp, bp, lp, rp, cv2.BORDER_CONSTANT, None, bg)
    return cv2.resize(img,(size,size))



def find_largest_feature(inp_img, scan_tl=None, scan_br=None):

    img = inp_img.copy()  
    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    # Loop through the image
    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
        
            if img.item(y, x) == 255 and x < width and y < height:  
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:  
                    max_area = area[0]
                    seed_point = (x, y)

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8)  

    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:  
                cv2.floodFill(img, mask, (x, y), 0)


            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    boundry = [[left, top], [right, bottom]]
    return img, np.array(boundry, dtype='float32'), seed_point



def extraction(image,square,size):
    
    digit = generate_digits(image,square)
    h,w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)

    _,boundry,seed = find_largest_feature(digit,[margin,margin],[w - margin, h - margin])
    digit = generate_digits(digit,boundry)

    w = boundry[1][0] - boundry[0][0]
    h = boundry[1][1] - boundry[0][1]

    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scaling(digit,size,6)
    else:
        return np.zeros((size,size),np.uint8)


def get_digits(img,squares,size):
    digits = []
    img = process_image(img.copy(),skip_dilate=True)
    for square in squares:
        digits.append(extraction(img,square,size))

    return digits



def getimageformpc(path):
    image = readimg(path = path)
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    processed_image = process_image(gray)
    corners_of_largest_poly = get_points(processed_image)
    # showimg(processed_image)
    cropped_image = image_cropper(gray , corners_of_largest_poly)
    cropped_image_colored = image_cropper(image , corners_of_largest_poly)


    # plotpoints(image,3,corners_of_largest_poly,square = False)

    squares,pones = reduceintopieces(cropped_image)

    # plotpoints(cropped_image,2,squares,square=True)

    digits = get_digits(cropped_image,squares,28)
    final_image = show_digits(digits)

    return np.array(final_image),np.array(pones),image,cropped_image_colored

    return final_image
# if __name__ == '__main__':
#     fromimage('img/one.png')
    