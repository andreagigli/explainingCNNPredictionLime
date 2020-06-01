#!/usr/bin/env python

import argparse

import os, sys

import numpy as np

from matplotlib import pyplot as plt

import keras
import keras.preprocessing.image as kerasimg
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries

import copy

from sklearn.metrics import pairwise_distances, ranking
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.feature_selection import SelectFromModel, RFE






###############################################################################
################################ IMAGE FUNCTIONS ##############################
###############################################################################
def load_images(image_path):
    # load the image and resize it to the dim required by the CNN
    img = kerasimg.load_img(image_path, target_size=(224, 224)) # (w,h,c) with c='RGB'
    img = kerasimg.img_to_array(img, dim_ordering='tf') / 255. # values in [0,1]
    #    img = np.expand_dims(img, axis=0) # (w,h,c) -> (i,w,h,c) to index multiple images?
    return img

def preprocess_image_theano_vgg16(img):
    assert len(img.shape)<=4, 'preprocess_image_theano_vgg6 accepts rbg images or collection of images'
    img = np.copy(img)
    if len(img.shape)<4:
        # cwh -> icwh
        img = np.expand_dims(img, axis=0)
    # iwhc -> icwh
    img = np.transpose(img,(0,3,1,2))
    # c: 'RGB'->'BGR'
    img = img[:, ::-1, :, :]
    # values in 0,255
    img *= 255.
    # Zero-center by mean pixel
    img[:, 0, :, :] -= 103.939 #normalize blue
    img[:, 1, :, :] -= 116.779 #normalize green
    img[:, 2, :, :] -= 123.68 #normalize red
    return img




######################################################################################
################################ EXPLANATION FUNCTIONS ###############################
######################################################################################
def take_top_labels(predictions, n_top_pred):
    labels_of_interest = decode_predictions(predictions, top=n_top_pred)[0]
    labels_of_interest_indices = (predictions.argsort()[:, -n_top_pred:])[0, ::-1]
    for i in range(len(labels_of_interest)):
        labels_of_interest[i] = labels_of_interest[i] + (labels_of_interest_indices[i],)
    print('Labels of interest (top predictions):', labels_of_interest)
    return labels_of_interest, labels_of_interest_indices


def generate_neighbors(segments, n_neighbors):
    # Determine (randomly) which superpixels will be "active" in each neighbor.
    # note: neighbor_brief_list consists of n_neighbors binary vectors of size (n_segments x 1) and represents
    # the explanatory representation (bool vector of on/off superpixels) of the neighbors of our image
    n_segments = np.unique(segments).shape[0]
    neighbor_brief_list = np.random.randint(0, 2, n_neighbors * n_segments).reshape((n_neighbors, n_segments))
    return neighbor_brief_list


def classify_neighbors(img, neighbor_brief_list, segments, model):
    # Create the image of the neighbors and classify them with the original classifier

    # # "screen" the off-superpixels in the neighbors.
    # superpix_avg_color = img.copy()
    # superpix_avg_color[:,:,:] = (0,0,0)

    # neigh image creation and classification are not sequential. To save memory, we launch a classification after the
    # creation of batches of neighbor images.
    neighbor_labels = []
    neighbor_img_list = []
    for neighbor_brief in neighbor_brief_list:
        neighbor = copy.deepcopy(img)
        # Determine a mask indicating the "off" superpixels of the neighbor
        mask_superpixels_off = np.zeros(segments.shape).astype(bool)
        for i in np.where(neighbor_brief == 0)[0]:
            mask_superpixels_off[segments == i] = True
        # Replace the off-superpixels with plain color
        # neighbor[mask_superpixels_off] = superpix_avg_color[mask_superpixels_off]
        neighbor[mask_superpixels_off] = (0,0,0)
        neighbor_img_list.append(neighbor)
        # At each batch_size neighbors, require the classification of the neighbors
        if len(neighbor_img_list) == 500:
            neighbor_labels.extend(model.predict(preprocess_image_theano_vgg16(np.array(neighbor_img_list))))
            neighbor_img_list = []
    # Classify the remaining neighbors
    if len(neighbor_img_list) > 0:
        neighbor_labels.extend(model.predict(preprocess_image_theano_vgg16(np.array(neighbor_img_list))))
        neighbor_img_list = []
    return np.array(neighbor_labels, ndmin=2)


def compute_neigh_relevance(x, neighbor_list, distance_metrics, kernel_width):
    # Compute the distance between each neighbor and x according to the specified metrics.
    # The distance is then "passed through" a Gaussian kernel function (output in [0,1]) to obtain the relevance,
    # which is inversely proportional to the distance.
    distances = pairwise_distances(
        neighbor_list,
        x.reshape(1, -1),
        metric=distance_metrics
    ).ravel()
    if kernel_width is None:
        kernel_width = np.percentile(distances, 50)
    neighbor_relevance = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
    if len(neighbor_relevance.shape) < 2:
        neighbor_relevance = neighbor_relevance[None]
    if neighbor_relevance.shape[1] != 1:
        neighbor_relevance = np.transpose(neighbor_relevance)
    return neighbor_relevance


def feature_selection(X, y, distances, feature_selection_criterion, n_desired_explanatory_features):
    # feature_selection_criterion = 'f_regression', 'mutual_info_regression', 'rfe', 'lassoCV'
    top_pos_feats_idx = 0
    top_neg_feats_idx = 0
    if n_desired_explanatory_features > X.shape[1]:
        n_desired_explanatory_features = X.shape[1]
        print('The number of desired explanatory superpixels is greater that the total number of superpixels!')

    if feature_selection_criterion == 'lasso':
        # The loss pi_i*|f(z_i)-g(z_i^i)|^2 corresponds to |f(sqrt(pi_i)z_i)-g(sqrt(pi_i)z_i^i)|^2
        # therefore it is sufficient to premultiply both instances and labels by pi_i and then perform
        # standard lasso. LassoCV automatically performs hyperparameter optimization. Lasso trains a sparse
        # linear regressor. The sparsity of the model's weights indicates that only the weights associated
        # to the most relevant features are different from 0. The weights can be both positive and negative
        # and their magnitude relates to their "influence" in the regression task. The most positive
        # weights correspond to the features that explain the ground labels, the most negative weights to those
        # features that are "contrary" to the ground labels.
        # NOTE: To use Lasso for feature selection one needs a high number of training instances (>1000 in this case)
        # otherwise all the weights will be zero and the results will be randomical!

        # Weight the neighbors by their relevances. (Premultiply x_i and y_i by pi_i).
        # TODO check distances' dimension
        X = X.astype(np.float64)
        tmp = np.sqrt(distances)
        X *= tmp
        y *= tmp

        # Train the lasso model and do feature selection
        model_weights = LassoCV().fit(X,y.ravel()).coef_
        if np.count_nonzero(model_weights)==0:
            print('Warning: LassoCV was not able to determine what are the informative superpixels (too few neighbors), imposing suboptimal small alpha')
            model_weights = LassoCV(alpha=0.00001).fit(X,y.ravel()).coef_
        tmp = np.argsort(model_weights)
        top_pos_feats_idx = (tmp[-n_desired_explanatory_features:])[::-1] # weight idx, from higher to lower weight
        top_neg_feats_idx = tmp[:n_desired_explanatory_features] # weight idx, from lower (higher in magnitude) to higher weight
        # do not consider superpixels associated to weights=0
        top_pos_feats_idx = top_pos_feats_idx[model_weights[top_pos_feats_idx].nonzero()]
        top_neg_feats_idx = top_neg_feats_idx[model_weights[top_neg_feats_idx].nonzero()]

    if feature_selection_criterion == 'rfe':

        # Weight the neighbors by their relevances. (Premultiply x_i and y_i by pi_i).
        # TODO check distances' dimension
        X = X.astype(np.float64)
        tmp = np.sqrt(distances)
        X *= tmp
        y *= tmp

        # Recursive feature elimination with Ridge Regressor
        model = RidgeCV()
        rfe = RFE(estimator=model, n_features_to_select=1)
        rfe.fit(X,y.ravel())
        ranking = rfe.ranking_
        tmp = np.argsort(ranking)
        top_pos_feats_idx = (tmp[-n_desired_explanatory_features:])[::-1] # features idx, from higher-ranked to lower-ranked features
        top_neg_feats_idx = tmp[:n_desired_explanatory_features] # features idx, from lower-ranked () to higher-ranked features # MAYBE THIS IS CONCEPTUALLY WRONG..

    return top_pos_feats_idx, top_neg_feats_idx


def mark_explanation_on_image(img, segments, top_pos_segments, top_neg_segments = None, only_pos_segments = False):
    img = img.copy()

    for seg in top_pos_segments:
        img[segments == seg, 1] = 1 # increase green level
    # plt.imshow(mark_boundaries(img, segments == top_pos_segments, outline_color=(0,1,0)))

    if only_pos_segments == False:
        for seg in top_neg_segments:
            img[segments == seg, 0] = 1 # increase red level
    # plt.imshow(mark_boundaries(img, segments == top_neg_segments, (1,0,0)))

    return img


def display_explanations(img, expl_image_list, top_predictions):

    l = len(expl_image_list)
    if l > 5:
        print("Warning: the explanations for only 5 labels have been shown")

    if l == 1:
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].set_xticklabels([])
        axes[0].set_yticklabels([])
        axes[1].imshow(expl_image_list[0])
        axes[1].set_title(str(top_predictions[0][3])+'   '+str(top_predictions[0][1]))
        axes[1].set_xticklabels([])
        axes[1].set_yticklabels([])
    elif l == 2:
        fig, axes = plt.subplots(1,3)
        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].set_xticklabels([])
        axes[0].set_yticklabels([])
        axes[1].imshow(expl_image_list[0])
        axes[1].set_title(str(top_predictions[0][3])+'   '+str(top_predictions[0][1]))
        axes[1].set_xticklabels([])
        axes[1].set_yticklabels([])
        axes[2].imshow(expl_image_list[1])
        axes[2].set_title(str(top_predictions[1][3])+'   '+str(top_predictions[1][1]))
        axes[2].set_xticklabels([])
        axes[2].set_yticklabels([])
    elif l == 3:
        fig, axes = plt.subplots(2,2)
        axes[0,0].imshow(img)
        axes[0,0].set_title("Original Image")
        axes[0,0].set_xticklabels([])
        axes[0,0].set_yticklabels([])
        axes[0,1].imshow(expl_image_list[0])
        axes[0,1].set_title(str(top_predictions[0][3])+'   '+str(top_predictions[0][1]))
        axes[0,1].set_xticklabels([])
        axes[0,1].set_yticklabels([])
        axes[1,0].imshow(expl_image_list[1])
        axes[1,0].set_title(str(top_predictions[1][3])+'   '+str(top_predictions[1][1]))
        axes[1,0].set_xticklabels([])
        axes[1,0].set_yticklabels([])
        axes[1,1].imshow(expl_image_list[2])
        axes[1,1].set_title(str(top_predictions[2][3])+'   '+str(top_predictions[2][1]))
        axes[1,1].set_xticklabels([])
        axes[1,1].set_yticklabels([])
    elif l == 4:
        fig = plt.subplots()
        ax00 = plt.subplot2grid((2,3),(0,0))
        ax00.imshow(img)
        ax00.set_title("Original Image")
        ax00.set_xticklabels([])
        ax00.set_yticklabels([])
        ax01 = plt.subplot2grid((2,3),(0,1))
        ax01.imshow(expl_image_list[0])
        ax01.set_title(str(top_predictions[0][3])+'   '+str(top_predictions[0][1]))
        ax01.set_xticklabels([])
        ax01.set_yticklabels([])
        ax02 = plt.subplot2grid((2,3),(0,2))
        ax02.imshow(expl_image_list[1])
        ax02.set_title(str(top_predictions[1][3])+'   '+str(top_predictions[1][1]))
        ax02.set_xticklabels([])
        ax02.set_yticklabels([])
        ax10 = plt.subplot2grid((2,3),(1,0))
        ax10.imshow(expl_image_list[2])
        ax10.set_title(str(top_predictions[2][3])+'   '+str(top_predictions[2][1]))
        ax10.set_xticklabels([])
        ax10.set_yticklabels([])
        ax11 = plt.subplot2grid((2,3),(1,1))
        ax11.imshow(expl_image_list[3])
        ax11.set_title(str(top_predictions[3][3])+'   '+str(top_predictions[3][1]))
        ax11.set_xticklabels([])
        ax11.set_yticklabels([])
    else:
        fig, axes = plt.subplots(2,3)
        axes[0,0].imshow(img)
        axes[0,0].set_title("Original Image")
        axes[0,0].set_xticklabels([])
        axes[0,0].set_yticklabels([])
        axes[0,1].imshow(expl_image_list[0])
        axes[0,1].set_title(str(top_predictions[0][3])+'   '+str(top_predictions[0][1]))
        axes[0,1].set_xticklabels([])
        axes[0,1].set_yticklabels([])
        axes[0,2].imshow(expl_image_list[1])
        axes[0,2].set_title(str(top_predictions[1][3])+'   '+str(top_predictions[1][1]))
        axes[0,2].set_xticklabels([])
        axes[0,2].set_yticklabels([])
        axes[1,0].imshow(expl_image_list[2])
        axes[1,0].set_title(str(top_predictions[2][3])+'   '+str(top_predictions[2][1]))
        axes[1,0].set_xticklabels([])
        axes[1,0].set_yticklabels([])
        axes[1,1].imshow(expl_image_list[3])
        axes[1,1].set_title(str(top_predictions[3][3])+'   '+str(top_predictions[3][1]))
        axes[1,1].set_xticklabels([])
        axes[1,1].set_yticklabels([])
        axes[1,2].imshow(expl_image_list[4])
        axes[1,2].set_title(str(top_predictions[4][3])+'   '+str(top_predictions[4][1]))
        axes[1,1].set_xticklabels([])
        axes[1,1].set_yticklabels([])

    plt.show()
    return fig




######################################################################################
######################################## MAIN ########################################
######################################################################################
def main():
    ### Parse inputs
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pathimage', required=True, help='path of the image to be classified and explained.')
    parser.add_argument('--ntoppred', required=True, type=int, help='number of top predictions to be explained.')
    parser.add_argument('--nneighbors', required=True, type=int, help='number of neighbors of img to be generated to explain the model.')
    parser.add_argument('--relevancekernelwidth', required=False, type=float, default=None, help='parameter of the kernel used for estimating neighbor relevance. Must range in [0,1]. Putting it very high corresponds in considering all the neighbors the same.')
    parser.add_argument('--featselectioncriterion', required=False, choices=['lasso','ref'], default='lasso', help='criterion used for feature selection.')
    parser.add_argument('--save', required=False, help='output path.')
    args = parser.parse_args()

    pathimage = args.pathimage
    n_top_pred = args.ntoppred
    n_neighbors = args.nneighbors
    feature_selection_criterion = args.featselectioncriterion
    relevance_kernel_width = args.relevancekernelwidth
    n_desired_explanatory_features = 3 # n most relevant superpixels to be shown in the final explanation
    out_path = args.save
    # pathimage = './data/breadpug.jpg'
    # n_top_pred = 3
    # n_neighbors = 500
    # feature_selection_criterion = 'lasso'
    assert n_neighbors>=3, 'Error: use more than 3 neighbors!'
    assert (n_top_pred in (1,2,3,4,5)), 'Please specify a number of desired top-predictions between 1 and 5a'
    # note: it is perfectly possible to have n_top_pred > 5 but the function display_explanations will not be able to
    # gather all the explanations in one plot. If you want n_top_pred > 5 modify the code in order to display each
    # explanation in a single plot.

    # Load image
    img = load_images(pathimage)


    ### Classify the object in the image
    model = VGG16(weights='imagenet', include_top=True)
    predictions = model.predict(preprocess_image_theano_vgg16(img))

    # Consider the first n_top_pred predictions
    labels_of_interest, labels_of_interest_indices = take_top_labels(predictions, n_top_pred)


    ### Explain each top prediction

    # Segment the image in superpixels
    segments = quickshift(img, kernel_size=5, max_dist=50, ratio=0.5)
    # plt.matshow(segments) # debug
    # plt.show() # debug

    # Generate the neighboroud of img. Each neighbor IS AN IMAGE which is obtained from img by RANDOMLY
    # "deactivating" some of its superpixels. "Deactivate" a superpixel means to replace the corresponding area
    # in img with plain color. We consider both a "brief" representation of the
    # neighbor, that is a boolean vector indicating which superpixels are active and which are inactive in the neighbor,
    # and a "img" neighbor representation, that is an image in which the inactive superpixels are black.
    neighbor_brief_list = generate_neighbors(segments, n_neighbors)
    neighbor_labels = classify_neighbors(img, neighbor_brief_list, segments, model)

    # Compute the distance between each neighbor and the original image
    neighbor_relevance = compute_neigh_relevance(neighbor_brief_list[0], neighbor_brief_list, 'cosine', relevance_kernel_width)

    expl_image_list = []
    for i in range(len(labels_of_interest_indices)): # for each top-prediction
        lab = labels_of_interest[i][-1]

        # Feature selection
        top_pos_feats, top_neg_feats = feature_selection( neighbor_brief_list, np.transpose((neighbor_labels[:, lab])[None]), neighbor_relevance, feature_selection_criterion, n_desired_explanatory_features)

        # "Impress" the explanation on the original image
        expl_image_list.append(mark_explanation_on_image(img, segments, top_pos_feats, top_neg_feats))

    # gather the explanations of all the desired labels (up to 5) in a single plot
    fig = display_explanations(img, expl_image_list, labels_of_interest)

    # Save the output
    if args.save is not None:
        fig.savefig(out_path)

    plt.show(fig) # warning: blocking call!


if __name__ == '__main__':
    main()
