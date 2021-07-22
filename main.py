import numpy as np
import cv2 as cv
import random
#The number of frames to overlap, modify
NUM_FRAMES = 4

# Alpha value for linear blending
ALPHA = 1/NUM_FRAMES

#You can predefine angles or random
ANGLES_RANDOM = True

# add predefined angles here, make sure they are same number as num frames
ANGLES_PREDEFINED = [90, 180, 45]

ANGLES = [random.randint(0, 360) for _ in range(NUM_FRAMES)] if ANGLES_RANDOM else ANGLES_PREDEFINED


def blend_imgs(imgs):
    '''
    Blends imgs together using linear blend ie. g(x) = alpha * (f_1(x) + f_2(x) + ... + f_n(x))
    Arguments:
        imgs:  list of numpy arrays of H x W x C ie. H- height, W - Width, C - Channels
    Returns:
        fused_img: blended image using linear blend
    '''
    fused_img = np.zeros_like(imgs[0].shape)
    for img in imgs:
        fused_img = np.uint8(fused_img + ALPHA*img) #running mean
    return fused_img


def get_affine_mat(angle, x_tran, y_tran):
    '''
    Rotates images at predefined origin (x_tran, y_tran). By doing
    translate custom origin to true origin, rotate image, translate true origin to custom origin
    Arguments:
        angle: angle to apply 2D rotation
        x_tran: x coordinate of origin
        y_tran: y coordinate of origin
    Returns:
        mat: 2x3 composited affine matrix
    '''
    #Convert angle from deg to rad
    angle = np.radians(-angle)
    rot = np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    tran_pre = np.array([[1, 0, -x_tran], [0, 1, -y_tran], [0, 0, 1.]])
    tran_post = np.array([[1, 0, x_tran], [0, 1, y_tran], [0, 0, 1.]]) # needed to be float numpy makes float if decimal point is included at least once in the array

    mat = tran_post @ rot @ tran_pre

    # turn 3x3 to 2 x 3
    mat = mat[0:2]

    #print(mat) # for debugging purposes
    return mat



def affine_transform_blend_imgs(imgs):
    '''
    Performs translation and rotation of list of images and blends together
    Arguments:
        imgs: list of numpy arrays each of shape H x W x C ie. H- height, W - Width, C - Channels
    Returns:
        blended_img: blended frame to be shown using opencv
    '''
    warped_imgs = []

    for img, angle in zip(imgs, ANGLES):


        height, width, channels = img.shape

        # PARAMETERS TO CHANGE
        x_tran = img.shape[1]/2 + img.shape[1]/5
        y_tran = img.shape[0]/2 - img.shape[0]/5

        aff = get_affine_mat(angle, x_tran , y_tran)
        warped_img = cv.warpAffine(img, aff, (img.shape[1], img.shape[0]))
        warped_imgs.append(warped_img)
    return blend_imgs(warped_imgs)

'''
Main code streams video feed using opencv and processes each frame, frame-by-frame.
'''
if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        #Copy frame NUM_FRAME times eqivalent to [frame, frame, ...] in simple list
        frames = [frame] * NUM_FRAMES

        #print(len(frames), ANGLES) # to debug
        # Perform rotation, translation and blending
        new_frame = affine_transform_blend_imgs(frames)
        #show frame
        cv.imshow('frame', new_frame)

        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
