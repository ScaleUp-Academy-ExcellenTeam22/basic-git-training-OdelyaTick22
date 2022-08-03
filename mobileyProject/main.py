from numpy import array

try:
    import os
    import json
    import glob
    import argparse
    import cv2
    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage import maximum_filter
    import typing
    from PIL import Image, ImageDraw

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise

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


# def find_tfl_lights(c_image: np.ndarray, **kwargs):
#     """
#     Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
#     :param c_image: The image itself as np.uint8, shape of (H, W, 3)
#     :param kwargs: Whatever config you want to pass in here
#     :return: 4-tuple of x_red, y_red, x_green, y_green
#     """
#     return [500, 510, 520], [500, 500, 500], [700, 710], [500, 500]


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


def search_by_color(image: np.ndarray, color: str) -> (array, typing.List):
    """
    Highlights the relevant color, convolutes the image and filter the list of attentions from it.
    :param image: The image that transferred for processing.
    :param color: The color of the attentions that should be found.
    :return: The processed image and the list of attentions.
    """
    if color == "red":
        img_color = image[:, :, 0]
        con = sg.convolve(img_color, kernel, mode='same')
        lst = np.argwhere(maximum_filter(con, 5) > 7000)
    else:
        print("Almog")
        img_color = image[:, :, 1]
        con = sg.convolve(img_color, kernel, mode='same')
        lst = np.argwhere(maximum_filter(con, 30) > 7000)
    return con, lst


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
    #convolved_image, lst_red = search_by_color(image, "red")
    convolved_image, lst_green = search_by_color(image, "green")

    plt.figure(57)
    plt.clf()
    h = plt.subplot(111)
    plt.imshow(convolved_image)

    plt.figure(52)
    plt.clf()
    plt.subplot(111, sharex=h, sharey=h)
    plt.imshow(Image.open(image_path))
    # for i in lst_red:
    #     red_y, red_x = i
    #     plt.plot(red_x, red_y, 'ro', color='r', markersize=1)
    for i in lst_green:
        green_y, green_x = i
        plt.plot(green_x, green_y, 'ro', color='g', markersize=1)

    plt.figure(55)
    plt.clf()
    plt.subplot(111, sharex=h, sharey=h)
    plt.imshow(convolved_image > 7000)


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
    # flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    flist = [r"C:\Users\IMOE001\images\berlin_000455_000019_leftImg8bit.png"]
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
