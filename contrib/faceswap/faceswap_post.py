#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import cv2
from numpy import mat

# Define 106 feature landmarks.
FACE_POINTS = list(range(0, 33))
MOUTH_POINTS = list(range(52, 72))
RIGHT_BROW_POINTS = list(range(43, 52))
LEFT_BROW_POINTS = list(range(97, 106))
RIGHT_EYE_POINTS = list(range(33, 43))
LEFT_EYE_POINTS = list(range(87, 97))
NOSE_POINTS = list(range(72, 87))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS +
                NOSE_POINTS + MOUTH_POINTS)

# Points from the cover image to overlay on the base image. The convex hull of each element will be overlaid.
OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
                   NOSE_POINTS + MOUTH_POINTS,]

# Amount of blur to use during color correction, as a fraction of the pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.5

# The bias of GaussianBlur
BLUR_BIAS = 128

# Feather parameter(an odd number) can blur the edges of the selection, causing the edges to fade out.
FEATHER_AMOUNT = 15

def transform_from_points(base_points, cover_points):
    """
    Aligning faces with a procrustes analysis.
    Return an affine transformation [s * r | T] such that: sum ||s*r*p1,i + T - p2,i||^2 is minimized.

    Args:
        base_points: the featured points of the input image whose face will be swapped
        cover_points: the featured points of another input image which will provide a target face

    Returns:
        Return the complete transformation between two faces as an affine transformation matrix
    """
    base_points = base_points.astype(np.float64)
    cover_points = cover_points.astype(np.float64)

    c1 = np.mean(base_points, axis=0)
    c2 = np.mean(cover_points, axis=0)

    base_points -= c1
    cover_points -= c2

    s1 = np.std(base_points)
    s2 = np.std(cover_points)

    base_points /= s1
    cover_points /= s2

    # Solve the procrustes problem by substracting centroids, scaling by the standard deviation,
    # and then using the SVD to calculate the rotation. See the following for more details:
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    u, s, vt = np.linalg.svd(base_points.T * cover_points)

    # The r we seek is in fact the transpose of the one given by u * vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    r = (u * vt).T
    convert_matrix = np.vstack([np.hstack(((s2 / s1) * r, c2.T - (s2 / s1) * r * c1.T)), np.matrix([0., 0., 1.])])

    return convert_matrix

def warp_img(cover_img, convert_matrix, base_shape):
    """
    Different sizes and angles caused by the shooting Angle and distance or the image resolution difference may lead to
    a stiff swap-result. We also need to carry out affine transformation according to the facial feature regions
    calculated by both of them, so that the facial features of the two can overlap as much as possible.
    In this function, we provide affine transformation from the cover_img to base_img based on the base_shape
    and convert_matrix.

    The result of transform_from_points function can then be plugged into OpenCV’s cv2.warpAffine function to map the second image onto the first.

    Args:
        cover_img: the cover img
        convert_matrix: transformation matrix resulted from the transform_from_points function
        base_shape: the image shape of the base_img
    """
    output_im = np.zeros(base_shape, dtype=cover_img.dtype)
    cv2.warpAffine(cover_img,
                   convert_matrix[:2],
                   (base_shape[1], base_shape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def correct_colors(img1, img2, landmarks1):
    """
    The different skin tones and lighting between the two images create edge discontinuities in the covered area.
    This function aims to homogenize the image color by altering the img2's color to adapt the img1.

    Args:
        img1: the image whose color is seen as the target （In this project, it should be the base_img）
        img2: the image whose color will be changed (In this project, it should be the cover_img)
        landmarks1: the facial feature landmarks of img1
    """
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0)
    )
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    img1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
    img2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)
    # Avoid divide-by-zero errors:
    img2_blur += (BLUR_BIAS * (img2_blur <= 1.0)).astype(img2_blur.dtype)
    return (img2.astype(np.float64) * img1_blur.astype(np.float64) /
            img2_blur.astype(np.float64))

def draw_convex_hull(img, points, color):
    """
    Draw the convex hull based on the points.

    Args:
        img: the input image
        points: the featured points inferred by the face-landmarks detection model
        color: the color of the convex hull

    Returns:
        A convex hull
    """
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(img, points, color=color)

def get_face_mask(img, landmarks):
    """
    Get an image mask for the facial features including eyebrows, eyes, nose, and mouth
    After the image mask is applied to the original image, only the white part of the corresponding mask
    in the original image can be displayed, while the black part will not be displayed.
    Therefore, we can achieve image "clipping" through the image mask.
    It is defined to generate a mask for an image and a marker matrix that draws convex polygons:
    the area around the eye, and the area around the nose and mouth.
    It is then feathered by 11 (FEATHER_AMOUNT) pixels outside the edge of the mask to help hide discontinuous areas.

    Args:
        img: the input image
        landmarks: the landmarks matrix inferred by the face-landmarks detection model

    Returns:
        A facial features mask
    """
    img = np.zeros(img.shape[:2], dtype=np.float64)
    for group in OVERLAY_POINTS:
        draw_convex_hull(img, landmarks[group], color=1)
    img = np.array([img, img, img]).transpose((1, 2, 0))
    img = (cv2.GaussianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    img = cv2.GaussianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return img

def swap_face(base_landmarks, cover_landmarks, base_face, cover_face):
    """
    Args:
        base_landmarks:  the featured landmarks of the base image (matrix)
        cover_landmarks: the featured landmarks of the cover image (matrix)
        base_face: the input image whose face will be swapped
        cover_face: another input image which will provide a target face

    Returns:
        the result of swapped face
    """
    convert_matrix = transform_from_points(base_landmarks[ALIGN_POINTS], cover_landmarks[ALIGN_POINTS])
    face_mask = get_face_mask(cover_face, cover_landmarks)
    warped_mask = warp_img(face_mask, convert_matrix, base_face.shape)
    combined_mask = np.max([get_face_mask(base_face, base_landmarks), warped_mask], axis=0)
    warped_im2 = warp_img(cover_face, convert_matrix, base_face.shape)
    warped_corrected_im2 = correct_colors(base_face, warped_im2, base_landmarks)
    output_im = base_face * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    return cv2.imwrite("only_face_swap.jpg", output_im)


