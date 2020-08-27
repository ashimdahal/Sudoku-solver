from tensorflow.keras.models import load_model
import numpy as np
from cv2 import cv2

model = load_model('model.h5')

# frame_image = inputs.main('img/one.png')
grid = np.zeros([9,9])


def identify_number(image):

    image_resize = cv2.resize(image, (28,28))    
    image_resize_2 = image_resize.reshape(1,1,28,28)    

    pred = model.predict(image_resize_2 , verbose = 0)

    return np.argmax(pred)


def extract_number(sudoku):
    sudoku = cv2.resize(sudoku, (450,450))


    # split sudoku
    grid = np.zeros([9,9])
    for i in range(9):
        for j in range(9):
#            image = sudoku[i*50+3:(i+1)*50-3,j*50+3:(j+1)*50-3]
            image = sudoku[i*50:(i+1)*50,j*50:(j+1)*50]
#            filename = "images/sudoku/file_%d_%d.jpg"%(i, j)

            if (np.sum(image) > 78988):
               
                image = image / 255
                
                grid[i][j] = identify_number(image)
            else:
                grid[i][j] = 0
    return grid.astype(int)

# final = extract_number(frame_image)
# print(final)

# solve.solve(final)
# solve.print_board(final)
# cv2.imshow('',frame_image)
# cv2.waitKey(0)