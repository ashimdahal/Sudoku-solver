from cv2 import cv2
import numpy as np
def printdigit(digit,im,postition):
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    return cv2.putText(im, str(digit), tuple(postition.astype('int')), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    

def print_many_digits(digits,image,positions):

    for i in range(len(digits)):
        if digits[i] == 0:
            pass
        else:
            image = printdigit(digits[i],image,positions[i])
    cv2.imshow('Solved solution',image)
    
    # return image
def main():
    image = cv2.imread('img/one.png')
    pos =(120,120)
    imagetxt = printdigit(7,image,pos)
    cv2.imshow('',imagetxt)
    cv2.waitKey(0)



if __name__ == '__main__':
    main()