#!/usr/bin/env python

from os import listdir
from os.path import isfile, join, abspath
import argparse
import cv2
import numpy as np
import colorsys
from sklearn.cluster import KMeans
from natsort import natsorted
import csv
import Levenshtein
from heapq import heappush, heappop
import itertools

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", help = "Path to directory")
ap.add_argument("-w", "--write", help = "Write the output image")
ap.add_argument("-s", "--show", default = False, action='store_true', help = "Show the output image")
ap.add_argument("files", nargs='*')

ap.add_argument("-n", "--colors", default = 5, type = int, help = "Number of Skittle colors")
ap.add_argument("--min", default = 30, type = int, help = "Minimum Skittle radius in pixels")
ap.add_argument("--max", default = 34, type = int, help = "Maximum Skittle radius in pixels")
ap.add_argument("--distance", default = 38, type = int, help = "Minimum Skittle distance between centers in pixels")
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

# convert counts to a string we can get the Levenshtein distance
flavor_characters = ["S", "O", "L", "A", "G"]
flavor_string_order = [4, 0, 1, 2, 3] # put grape next to strawberry since they're likely to be misclassified
def to_flavor_string(result, order=flavor_string_order):
    return "".join([flavor_characters[i] * result[i] for i in order])

results = [] # (path, cluster_counts, flavor_string)
known_results = {}

# read the manual counts
with open('skittles/skittles.txt') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    next(reader) # skip header
    for index, row in enumerate(reader):
        counts = list(map(int, row))
        path = join("./skittles/images", "{}.jpg".format(index + 1))
        known_results[abspath(path)] = (path, counts, to_flavor_string(counts))

def process(path):
    # load the image and then convert it to grayscale
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

        flavor_string = to_flavor_string(cluster_counts)

        row = [str(count) for count in cluster_counts]

        # if we have known result for this path, append the Levenshtein distance to the row
        dist = None
        if abspath(path) in known_results:
            dist = Levenshtein.distance(flavor_string, known_results[abspath(path)][2])
            row.append(str(dist))

        # print("\t".join(row))

        results.append((path, cluster_counts, flavor_string, dist))

        if args["write"] or args["show"]:
            output = image.copy()

            # draw some stuff to validate it
            for index, (x,y,r) in enumerate(circles):
                cluster_index = predictions[index]

                color = kmeans.cluster_centers_[cluster_index]
                cv2.circle(output, (x, y), r-1, (255,255,255), 2)
                cv2.circle(output, (x, y), r+1, color, 3)

                label_text = str(cluster_index)
                label_size = cv2.getTextSize(label_text, FONT_FACE, 0.5, 2)[0]
                label_position = (int(x - label_size[0] / 2), int(y + label_size[1] / 2))
                cv2.putText(output, label_text, label_position, FONT_FACE, 0.5, (255,255,255), 2, cv2.LINE_AA)

            if args["write"]:
                cv2.imwrite(args["write"], output)

            if args["show"]:
                cv2.imshow("output", np.hstack([image, output]))
                cv2.waitKey(0)

DISTANCE_THREASHOLD = 2
# returns closest matching results
def get_best_matches(results, threshold = DISTANCE_THREASHOLD):
    heap = []
    for index1, (path1, _, string1, _) in enumerate(results):
        for index2, (path2, _, string2, _) in enumerate(results):
            if index1 < index2:
                # compute the Levenshtein distance and add to a heap queue
                dist = Levenshtein.distance(string1, string2)
                if dist <= threshold:
                    heappush(heap, (dist, path1, path2))
    return heap

def get_known_stats(results):
    distance_total = 0
    distance_max = max([d for (_,_,_,d) in results])
    distance_distribution = [0] * (distance_max + 1)
    for (_, _, _, distance) in results:
        distance_total += distance or 0
        distance_distribution[distance] += 1
    return (distance_total, distance_max, distance_distribution)

# determines the best order for flavor string by comparing to known results
def get_best_flavor_string_orders(results, known_results):
    heap = []
    for order in itertools.permutations(flavor_string_order):
        distance_total = 0
        for (path, counts, _, distance) in results:
            if distance > 0:
                distance_total += Levenshtein.distance(
                    to_flavor_string(counts, order),
                    to_flavor_string(known_results[abspath(path)][1], order)
                )
        heappush(heap, (distance_total, order))
    return heap

if args["files"]:
    for f in args["files"]:
        process(f)
else:
    directory = args["directory"] or "./skittles/images"
    for f in natsorted(listdir(directory)):
        path = join(directory, f)
        if isfile(path):
            process(path)

# log differences between known and actual results
(distance_total, distance_max, distance_distribution) = get_known_stats(results)
print("Known distance total:", distance_total)
print("Known distance max:", distance_max)
print("Known distance distribution:", distance_distribution)

best_matches = get_best_matches(results)
print ("Matches:")
while len(best_matches) > 0:
    print(heappop(best_matches))

# determine the best flavor string order for comparing with Levenshtein distance
# print(heappop(get_best_flavor_string_orders(results, known_results)))