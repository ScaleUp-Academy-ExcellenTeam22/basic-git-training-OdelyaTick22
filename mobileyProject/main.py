import numpy

try:
    import os
    import json
    import glob
    import argparse
    import cv2
    import numpy as np
    from scipy import signal as sg
    # from scipy.ndimage.filters import maximum_filter
    from scipy.ndimage import maximum_filter

    from PIL import Image, ImageDraw

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

# light = 1
# dark = -69/52
#
# kernel = np.asarray([[dark, dark, dark, dark, dark, dark, dark, dark, dark, dark, dark, dark, dark],
#                      [dark, dark, dark, dark, light, light, light, light, light, dark, dark, dark, dark],
#                      [dark, dark, dark, light, light, light, light, light, light, light, dark, dark, dark],
#                      [dark, dark, light, light, light, light, light, light, light, light, light, dark, dark],
#                      [dark, light, light, light, light, light, light, light, light, light, light, light, dark],
#                      [dark, light, light, light, light, light, light, light, light, light, light, light, dark],
#                      [dark, light, light, light, light, light, light, light, light, light, light, light, dark],
#                      [dark, light, light, light, light, light, light, light, light, light, light, light, dark],
#                      [dark, light, light, light, light, light, light, light, light, light, light, light, dark],
#                      [dark, dark, light, light, light, light, light, light, light, light, light, dark, dark],
#                      [dark, dark, dark, light, light, light, light, light, light, light, dark, dark, dark],
#                      [dark, dark, dark, dark, light, light, light, light, light, dark, dark, dark, dark],
#                      [dark, dark, dark, dark, dark, dark, dark, dark, dark, dark, dark, dark, dark]])
kernel = np.asarray(
    [[-69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58,
      -69 / 58],
     [-69 / 58, -69 / 58, -69 / 58, 1, 1, 1, 1, 1, 1, -69 / 58, -69 / 58, -69 / 58],
     [-69 / 58, -69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58, -69 / 58],
     [-69 / 58, -69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58, -69 / 58],
     [-69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58],
     [-69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58],
     [-69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58],
     [-69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58],
     [-69 / 58, -69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58, -69 / 58],
     [-69 / 58, -69 / 58, 1, 1, 1, 1, 1, 1, 1, 1, -69 / 58, -69 / 58],
     [-69 / 58, -69 / 58, -69 / 58, 1, 1, 1, 1, 1, 1, -69 / 58, -69 / 58, -69 / 58],
     [-69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58, -69 / 58,
      -69 / 58]])


def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding * 2, image.shape[1] + padding * 2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break
    return output


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    ### WRITE YOUR CODE HERE ###
    ### USE HELPER FUNCTIONS ###
    return [500, 510, 520], [500, 500, 500], [700, 710], [500, 500]


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    # show_image_and_gt(image, objects, fig_num)
    plt.figure(56)
    plt.clf()
    h = plt.subplot(111)
    plt.imshow(image)

    # img = cv2.imread(image_path)
    # r = img.copy()
    # r[:, :, 0] = 0
    # r[:, :, 1] = 0
    # img = r.convert('L')
    img=numpy.asarray(Image.open(image_path).convert('L'))
    convoluted_image = convolve2D(img, kernel)

    plt.figure(57)
    plt.clf()
    plt.subplot(111, sharex=h, sharey=h)
    plt.imshow(convoluted_image)

    plt.figure(59)
    plt.clf()
    plt.subplot(111, sharex=h, sharey=h)
    plt.imshow(convoluted_image > 5000)

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = "../images"

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
        # img = np.array(Image.open(image).convert('L'))
        # convolve2D(img, kernel)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
