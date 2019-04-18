from os import listdir
from os.path import isfile, join
import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
from natsort import natsorted

ap = argparse.ArgumentParser()
group = ap.add_mutually_exclusive_group(required=True)
group.add_argument("-d", "--directory", help = "Path to directory")
group.add_argument("-i", "--image", help = "Path to the image")
ap.add_argument("-w", "--write", help = "Write the output image")
ap.add_argument("-s", "--show", default = False, action='store_true', help = "Show the output image")

ap.add_argument("-n", "--colors", default = 5, type = int, help = "Number of Skittle colors")
ap.add_argument("--min", default = 24, type = int, help = "Minimum Skittle radius in pixels")
ap.add_argument("--max", default = 36, type = int, help = "Maximum Skittle radius in pixels")
ap.add_argument("--distance", default = 36, type = int, help = "Minimum Skittle distance between centers in pixels")
# advanced HoughCircles parameters:
ap.add_argument("--dp", default = 2, type = int, help = "Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.")
ap.add_argument("--p1", default = 50, type = int, help = "First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).")
ap.add_argument("--p2", default = 30, type = int, help = "Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.")
args = vars(ap.parse_args())

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_CLUSTER_CENTERS = np.asarray([
    [12.31981247,  10.43815736,  56.97093355], # Strawberry
    [16.83754665,  43.00224546, 130.91020123], # Orange
    [11.42766936, 104.58650005, 139.44836321], # Lemon
    [15.66960301,  75.41447827,  18.51117709], # Apple
    [14.77467167,  12.48393894,  17.08273171], # Grape
 ])

def process(path):
    # load the image, clone it for output, and then convert it to grayscale
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        args["dp"],
        args["distance"],
        param1=args["p1"],
        param2=args["p2"],
        minRadius=args["min"],
        maxRadius=args["max"]
    )

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        colors = []
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # create a mask over the circle
            mask = np.zeros((image.shape[0],image.shape[1]), np.uint8)
            cv2.circle(mask,(x,y),r,(255,255,255),-1)
            # get the average rgb value of the image with the mask applied
            rgb = cv2.mean(image, mask=mask)[0:3]
            colors.append(rgb)

        # cluster the rgb values using kmeans
        X = np.array(colors)
        is_default_colors = len(DEFAULT_CLUSTER_CENTERS) == args["colors"]
        kmeans = KMeans(
            n_clusters=args["colors"],
            init=DEFAULT_CLUSTER_CENTERS if is_default_colors else "random",
            n_init=1 if is_default_colors else 2
        ).fit(X)

        predictions = kmeans.predict(X)
        
        cluster_counts = [0] * args["colors"]
        for cluster_index in predictions:
            cluster_counts[cluster_index] += 1
        print("\t".join([str(count) for count in cluster_counts]))

        if args["write"] or args["show"]:
            output = image.copy()

            # draw some stuff to validate it
            for index, (x,y,r) in enumerate(circles):
                cluster_index = predictions[index]

                color = kmeans.cluster_centers_[cluster_index]
                cv2.circle(output, (x, y), r-1, (255,255,255), 2)
                cv2.circle(output, (x, y), r, color, 2)

                label_text = str(cluster_index)
                label_size = cv2.getTextSize(label_text, FONT_FACE, 0.5, 2)[0]
                label_position = (int(x - label_size[0] / 2), int(y + label_size[1] / 2))
                cv2.putText(output, label_text, label_position, FONT_FACE, 0.5, (255,255,255), 2, cv2.LINE_AA)

            if args["write"]:
                cv2.imwrite(args["write"], output)

            if args["show"]:
                cv2.imshow("output", np.hstack([image, output]))
                cv2.waitKey(0)

if args["directory"]:
    for f in natsorted(listdir(args["directory"])):
        path = join(args["directory"], f)
        if isfile(path):
            process(path)
elif args["image"]:
    process(args["image"])
