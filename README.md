# Facade_Storefront_Potential_Rater
takes in pictures of facades/street views, and returns a score indicating the potential of the building in the picture to be converted into a store front


1. featurize images (identify doors, windows...) with pix2pix
model used to train and predict: https://github.com/affinelayer/pix2pix-tensorflow
datasets used: any subset of {CMP facade, ECPFD, etrims, graz50_facade_dataset, ImageSets, labelmefacade, ParisArtDecoFacadesDataset}
data processing: standardized feature colouring, opt. colour scheme adjustments (ie. make the storefront a colour between a door and a window)
output (input for 2 & 3): images with identified features colour-coded

2. classify storefront or not on featurized images
input: featurized images from step 1, and corresponding label for storefront (1) or not (0)
training (storefront_classifier.py): uses a CNN to train the classifier, adjusting model setup and parameters along the way
sample model: model_baselinelmf350_32channels_slow_reexcite_fullcolour_p1.h5

3. rate facade/street view images
input: featurized images from step 1, trained model from step 2
predicting (classifier_prediction.py): takes a directory containing featurized images, and runs the classification model
output: a storefront potential score, between 0 and 1, printed for each images

* see Storefront_Potential_of_City_Facedes.pptm for a more vitualized presentation
