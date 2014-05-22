/*
     OpenSeqSLAM
     Copyright 2013, Niko Sünderhauf Chemnitz University of Technology niko@etit.tu-chemnitz.de

     OpenSeqSLAM is an open source Matlab implementation of the original SeqSLAM algorithm published by Milford and Wyeth at ICRA12 [1]. SeqSLAM performs place recognition by matching sequences of images.
*/

//
//  OpenSeqSLAM.h
//  OpenSeqSLAM
//
//  Created by Saburo Okita on 20/05/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//



#ifndef __OpenSeqSLAM__OpenSeqSLAM__
#define __OpenSeqSLAM__OpenSeqSLAM__

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class OpenSeqSLAM {
public:
    OpenSeqSLAM();
    void init( int patch_size, int local_radius, int matching_distance );
    
    
    Mat preprocess( Mat& image );
    vector<Mat> preprocess( vector<Mat>& images );
    Mat calcDifferenceMatrix( vector<Mat>& set_1, vector<Mat>& set_2 );
    
    Mat enhanceLocalContrast( Mat& diff_matrix, int local_radius = 10 );
    pair<int, double> findMatch( Mat& diff_mat, int N, int matching_dist );
    Mat findMatches( Mat& diff_mat, int matching_dist = 10 );
    
    Mat apply( vector<Mat>& set_1, vector<Mat>& set_2 );
    
protected:
    int patchSize;
    int localRadius;
    int matchingDistance;
    int RWindow;
    float minVelocity;
    float maxVelocity;
    Size imageSize;
    
    double convertToSampleStdDev( double pop_stddev, int N );
    Mat normalizePatch( Mat& image, int patch_size );
    
};

#endif /* defined(__OpenSeqSLAM__OpenSeqSLAM__) */
