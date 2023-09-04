import numpy as np
import cv2


def line_params(p1, p2):
    """
    p:(x, y), y = w * x + b
    :param p1: point 1
    :param p2: point 2
    :return: w, b
    """
    x1, y1 = p1
    x2, y2 = p2

    w = (y2 - y1) / (x2 - x1)
    b = (y1 * x2 - x1 * y2) / (x2 - x1)

    return w, b


def y_(x, w, b):
    y = w * x + b
    return y


def x_(y, w, b):
    x = (y - b) / w
    return x


def create_mask(orig_img, lines):
    img = cv2.imread(orig_img)
    mask = np.zeros_like(img)
    height, width, channel = mask.shape
    print(height, width)

    left, top, right, bottom = lines
    for x in range(width):
        for y in range(height):
            if x_(y, left[0], left[1]) <= x <= x_(y, right[0], right[1]) and \
                    y_(x, top[0], top[1]) <= y <= y_(x, bottom[0], bottom[1]):
                mask[y, x, 2] = 255
    return mask


def add_mask(orig_mask, lines):
    mask = cv2.imread(orig_mask)
    height, width, channel = mask.shape
    print(height, width)

    left, top, right, bottom = lines
    for x in range(width):
        for y in range(height):
            if x_(y, left[0], left[1]) <= x <= 500 and \
                    y_(x, top[0], top[1]) <= y <= y_(x, bottom[0], bottom[1]):
                mask[y, x, 2] = 255
    return mask


def main():
    # orig_img = '../images/records/vlcsnap-2019-08-02-16h01m50s221.png'
    # left = line_params((180, 210), (35, 370))
    # top = line_params((180, 210), (500, 330))
    # right = line_params((500, 330), (495, 479))
    # bottom = line_params((0, 479), (495, 479))

    # sawanini_1
    orig_img = cv2.imread('D:\\XIO_Safe_1_alarm_X\\images\masks\\'
                          'line_2.jpg')
    mask = np.zeros_like(orig_img)

    # points = np.array([[110, 253], [225, 196], [425, 235], [380, 362]],
    #                   dtype=np.int32)#line_2
    # points = np.array([[251, 80], [605, 71], [640, 129], [640, 210], [604, 228], [601, 315], [346, 454]],
    #                   dtype=np.int32)#line_3d
    points = np.array([[168, 309], [325, 362], [295, 456], [117, 382]],
                      dtype=np.int32)#line_4_1
    # points = np.array([[161, 174], [404, 480], [531, 480], [212, 168]],
    #                   dtype=np.int32)
    # points = np.array([[297, 58], [542, 87], [630, 480], [195, 480]],
    #                  dtype=np.int32) # lvd
    # points = np.array([[155, 135], [87, 398], [595, 399], [569, 132]],
    #                     dtype=np.int32) # m5000
    # points = np.array([[171, 323], [120, 386], [181, 410], [185, 393], [226, 410], [225, 427], [297, 452], [325, 370]],
    #                   dtype=np.int32)# line_4_2
    mask = cv2.fillPoly(mask, [points], (0, 80, 255))
    # baobantongyong
    # orig_img = '../images/records/vlcsnap-2019-09-03-09h50m51s942.png'
    # left = line_params((115, 77), (23, 479))
    # top = line_params((115, 77), (397, 93))
    # right = line_params((397, 93), (405, 479))
    # bottom = line_params((23, 479), (405, 479))
    # mask = create_mask(orig_img, [left, top, right, bottom])

    # orig_mask = '../images/masks/penfenshang.jpg'
    # left = line_params((180, 150), (68, 479))
    # top = line_params((180, 150), (445, 210))
    # right = line_params((445, 210), (495, 479))
    # bottom = line_params((65, 479), (490, 479))
    # mask = add_mask(orig_mask, [left, top, right, bottom])

    # orig_img = '../images/records/vlcsnap-2019-08-02-16h02m31s252.png'
    # left = line_params((100, 175), (45, 385))
    # top = line_params((100, 175), (170, 180))
    # right = None
    # bottom = line_params((45, 385), (490, 385))
    # # mask = create_mask(orig_img, [left, top, right, bottom])
    # orig_mask = '../images/masks/sawanini_2.jpg'
    # left = line_params((220, 100), (169, 188))
    # top = line_params((220, 115), (480, 115))
    # right = None
    # bottom = line_params((100, 175), (170, 183))
    # mask = add_mask(orig_mask, [left, top, right, bottom])

    cv2.imshow('mask', mask)

    cv2.imwrite('../images/masks/line_4_1.jpg', mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # img_path = '../images/records/vlcsnap-2019-08-02-16h02m06s098.png'
    # mask_path = '../images/masks/sawanini_2.jpg'
    #
    # img = cv2.imread(img_path)
    # mask = cv2.imread(mask_path)
    #
    # overlap = cv2.addWeighted(img, 1, mask, 0.6, 0)
    #
    # cv2.imshow('overlap', overlap)
    # cv2.waitKey(0)
    main()

