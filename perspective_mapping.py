import cv2
import numpy as np
import sys

def nothing(x):
    pass

# Create a black image, a window
img = cv2.imread(sys.argv[1])
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('alpha','image',0,360,nothing)
cv2.createTrackbar('beta','image',0,360,nothing)
cv2.createTrackbar('gamma','image',0,360,nothing)
cv2.createTrackbar('f','image',0,2000,nothing)
cv2.createTrackbar('z_translation','image',0,2000,nothing)
cv2.createTrackbar('x_translation','image',1000,2000,nothing)
cv2.createTrackbar('y_translation','image',1000,2000,nothing)

first = True
transformation_mat = np.zeros((3,3))

while(1):

    # get current positions of four trackbars
    alpha = cv2.getTrackbarPos('alpha','image') * np.pi/180
    beta = cv2.getTrackbarPos('beta','image') * np.pi/180
    gamma = cv2.getTrackbarPos('gamma','image') * np.pi/180
    f = cv2.getTrackbarPos('f','image')
    z_translation = cv2.getTrackbarPos('z_translation','image')
    x_translation = cv2.getTrackbarPos('x_translation','image')
    y_translation = cv2.getTrackbarPos('y_translation','image')

    h = img.shape[0]
    w = img.shape[1]

    # translate the image roughly to the center of the result
    image_translation_mat = np.array(
        [[1, 0, -w/2],
         [0, 1, -h*1.2],
         [0, 0, 0],
         [0, 0, 1]]
    )

    # build the rotation martix
    rot_X = np.array(
        [[1, 0,              0,             0],
         [0, np.cos(alpha), -np.sin(alpha), 0],
         [0, np.sin(alpha),  np.cos(alpha), 0],
         [0, 0,              0,             1]]
    )
    rot_Y = np.array(
        [[np.cos(beta), 0, -np.sin(beta), 0],
         [0,            1,  0,            0],
         [np.sin(beta), 0,  np.cos(beta), 0],
         [0,            0,  0,            1]]
    )
    rot_Z = np.array(
        [[np.cos(gamma), -np.sin(gamma), 0, 0],
         [np.sin(gamma),  np.cos(gamma), 0, 0],
         [0,              0,             1, 0],
         [0,              0,             0, 1]]
    )
    rotation = np.matmul(np.matmul(rot_X, rot_Y), rot_Z)

    # translation to the projected plane
    camera_translation = np.array(
        [[1, 0, 0, x_translation-1000],
         [0, 1, 0, y_translation-1000],
         [0, 0, 1, z_translation],
         [0, 0, 0, 1]]
    )

    # an estimated intrinsic calibration
    intrinsic = np.array(
        [[f, 0, w/2, 0],
         [0, f, h/2, 0],
         [0, 0, 1, 0]]
    )

    # M = ((K * (T * R)) * image_translation)
    extrinsic = np.matmul(camera_translation, rotation) # 4x4 * 4x4 -> 4x4
    transformation = np.matmul(intrinsic, extrinsic) # 3x4 * 4x4 -> 3x4
    transformation_mat = np.matmul(transformation, image_translation_mat) # 3x4 * 4x3 -> 3x3  

    dst = cv2.warpPerspective(img, transformation_mat, (w,int(h*1.5)), flags=(cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP))
    
    cv2.imshow('image',dst)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == 113: # Esc or q
        break
    elif k == 112: # p
        print("---------PRINT OUT ---------")
        print("rotation\n", rotation)
        print("camera_translation\n", camera_translation)
        print("extrinsic\n", extrinsic)
        print("intrinsic\n", intrinsic)
        print("transformation_mat\n", transformation_mat)

cv2.destroyAllWindows()