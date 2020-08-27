import getinput as Sboard
import predict as predictor
import textsolver as solve
import arimage as ar
from cv2 import cv2
import numpy as np
import time

def solver(board):
    board_numbers = predictor.extract_number(board)
    known_numbers = board_numbers.copy()
    solve.solve(board_numbers)
    return board_numbers,board_numbers- known_numbers

def getimageformpc(path='img/three.png'):
    soduku_board,squares,original,cropped_image = Sboard.getimageformpc(path)
    return soduku_board,squares,original,cropped_image

def solveFromVideo():
    vid = cv2.VideoCapture(0)
    while True:
        suc,image= vid.read()
        image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = image[150:,120:]
        processed_image = Sboard.process_image(image)
        corners_of_largest_poly = Sboard.get_points(processed_image)
        cropped_image = Sboard.image_cropper(image , corners_of_largest_poly)
        squares,pones = Sboard.reduceintopieces(cropped_image)
        digits = Sboard.get_digits(cropped_image,squares,28)
        soduku_board = Sboard.show_digits(digits)
        
        
        cv2.imshow('usee',image)

        # cv2.imshow('cropped_image',cropped_image)
    

        answer,kn=solver(soduku_board)
        print(answer)
        cim  = cropped_image.copy()
        ar.print_many_digits(kn.reshape(-1),cim,pones)
        cv2.imshow('extracted after feature extraction',soduku_board)
        cv2.imshow('cropped_video',cropped_image)
        
        cv2.waitKey(50)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
def solvefrompc():
    imagelink = ['img/one.png','img/two.png','img/three.jpg']
    Extracted_board,squares,img,cropped_image=  getimageformpc(imagelink[2])
    answer,kn = solver(Extracted_board)
    solve.print_board(answer)
    # answer = np.array(answer)
    cim  = cropped_image.copy()
    ar.print_many_digits(kn.reshape(-1),cim,squares)
    cv2.imshow('extracted image after final feature extraction',Extracted_board)
    cv2.imshow('cropped_image feature extraction number one',cropped_image)
    cv2.imshow('input_image by user',img)
    cv2.waitKey(0)

if __name__ == '__main__':
    solvefrompc()


