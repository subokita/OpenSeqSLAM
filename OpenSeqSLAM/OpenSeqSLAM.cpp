//
//  OpenSeqSLAM.cpp
//  OpenSeqSLAM
//
//  Created by Saburo Okita on 20/05/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include "OpenSeqSLAM.h"


OpenSeqSLAM::OpenSeqSLAM(){
    patchSize           =  8;
    localRadius         = 10;
    matchingDistance    = 10;
    imageSize           = Size(64, 32);
    minVelocity         = 0.8;
    maxVelocity         = 1.2;
    RWindow             = 10;
}

void OpenSeqSLAM::init( int patch_size, int local_radius, int matching_distance ) {
    this->patchSize         = patch_size;
    this->localRadius       = local_radius;
    this->matchingDistance  = matching_distance;
}

/**
 * Convert to sample std deviation, since
 * OpenCV's meanstddev uses population's std deviation
 */
double OpenSeqSLAM::convertToSampleStdDev( double pop_stddev, int N ) {
    return sqrt( (pop_stddev * pop_stddev * N) / (N - 1) );
}

/**
 * Normalize each local patches in the image
 */
Mat OpenSeqSLAM::normalizePatch( Mat& image, int patch_size ) {
    Mat result = image.clone();
    Mat patch, patch_mean, patch_stddev, temp;
    
    int patch_area = patch_size * patch_size;
    
    for(int y = 0; y < result.rows; y+= patch_size ) {
        for(int x = 0; x < result.cols; x+= patch_size ) {
            /* Extract patch */
            patch = Mat( result, Rect(x, y, patch_size, patch_size) );
            
            
            /* Find the mean and std dev, calc  */
            meanStdDev( patch, patch_mean, patch_stddev );
            double mean_val   = patch_mean.at<double>(0, 0);
            double stddev_val = convertToSampleStdDev( patch_stddev.at<double>(0, 0), patch_area );
            
            
            /* Well, to avoid buffer issues, let's use double for this iteration */
            patch.convertTo( temp, CV_64FC1 );
            
            /* In Matlab 127 + x / 0.0 == 0, while in here, 127 + x / 0.0 == 127, so we need to handle that case  */
            if( stddev_val > 0.0 ) {
                
                /* Normalize the patch */
                for( MatIterator_<double> itr = temp.begin<double>(); itr != temp.end<double>(); itr++ )
                    *itr = 127 + cvRound( (*itr - mean_val) / stddev_val );
            }
            else
                temp = Scalar::all(0);
            
            temp.convertTo( patch, CV_8UC1 );
        }
    }
    
    return result;
}

/**
 * Preprocess stage, includes
 * - conversion to grayscale
 * - resizing to 32 x 64
 * - normalize local patches of size 8 x 8
 */
Mat OpenSeqSLAM::preprocess( Mat& image ) {
    Mat result = image.clone();
    
    if ( result.channels() > 1 )
        cvtColor( result, result, CV_BGR2GRAY );
    
    if( result.cols != imageSize.width && result.rows != imageSize.height )
        resize( result, result, imageSize, INTER_LANCZOS4 );
    
    return normalizePatch( result, this->patchSize );
}

/**
 * Perform preprocessing on a set of images
 */
vector<Mat> OpenSeqSLAM::preprocess( vector<Mat>& images ) {
    vector<Mat> result;
    for( Mat& image: images )
        result.push_back( preprocess( image ) );
    return result;
}

/**
 * Calculate the difference matrix between two set of images
 */
Mat OpenSeqSLAM::calcDifferenceMatrix( vector<Mat>& set_1, vector<Mat>& set_2 ) {
    int n = static_cast<int>(set_1.size());
    int m = static_cast<int>(set_2.size());
    
    Mat diff_mat = Mat::zeros(n, m, CV_32FC1 );
    Mat mat_1, mat_2;
    
    for( int i = 0; i < n; i++ ) {
        float * diff_ptr = diff_mat.ptr<float>(i);
        
        /* Difference is between two images is calculated as */
        /* the average sum of absolute difference between them */
        for( int j = 0; j < m; j++ )
            diff_ptr[j] = sum( abs(set_2[j] - set_1[i]) )[0] / n;
    }
    
    return diff_mat;
}

/**
 * Enhance the contrast for the local region of the difference matrix (specified within the local radius)
 * New contrast = (diff_matrix - mean) / stddev
 */
Mat OpenSeqSLAM::enhanceLocalContrast( Mat& diff_matrix, int local_radius ) {
    int rows = diff_matrix.rows;
    int cols = diff_matrix.cols;
    
    Mat enhanced( diff_matrix.size(), diff_matrix.type(), Scalar(0) );
    
    Mat patch_mean  (1, cols, CV_32FC1, Scalar(0)),
        patch_stddev(1, cols, CV_32FC1, Scalar(0));
    
    for( int i = 0; i < rows; i++ ) {
        /* Get the local patch from the difference matrix */
        int lower_bound = MAX( 0, i - local_radius / 2 );
        int upper_bound = MIN( rows, i + 1 + local_radius / 2 );
        Mat local_patch = diff_matrix.rowRange( lower_bound, upper_bound );
        
        /* Calculate the mean and std dev for each patch */
        float * mean_ptr     = patch_mean.ptr<float>(0);
        float * stddev_ptr   = patch_stddev.ptr<float>(0);
        for( int j = 0; j < cols; j++ ){
            Mat temp_mean, temp_stddev;
            meanStdDev( local_patch.col(j), temp_mean, temp_stddev );
            
            mean_ptr[j]   = temp_mean.at<double>(0, 0);
            stddev_ptr[j] = convertToSampleStdDev( temp_stddev.at<double>(0, 0), local_patch.rows );
        }
        
        /* Enhance contrast by (local_patch - patch_mean) / patch_stddev */
        enhanced.row(i) = (diff_matrix.row(i) - patch_mean).mul( 1./patch_stddev );
    }
    
    /* Shift so that the minimum value in the matrix is 0 */
    double min_val;
    minMaxLoc( enhanced, &min_val );
    enhanced = enhanced - min_val;
    
    return enhanced;
}

/**
 * Given the difference matrix, and N index, find the image that
 * has a good match within the matching distance from image N
 * This method returns the matching index, and its score
 */
pair<int, double> OpenSeqSLAM::findMatch( Mat& diff_mat, int N, int matching_dist ) {
    int move_min = static_cast<int>( minVelocity * matching_dist);
    int move_max = static_cast<int>( maxVelocity * matching_dist);
    
    /* Matching is based on max and min velocity */
    Mat velocity( 1, move_max - move_min + 1, CV_64FC1 );
    double * velocity_ptr = velocity.ptr<double>(0);
    for( int i = 0; i < velocity.cols; i++ )
        velocity_ptr[i] = (move_min + i * 1.0) / matching_dist;
    velocity = velocity.t();
    
    
    /* Create incremental indices based on the previously calculated velocity */
    Mat increment_indices( move_max - move_min + 1, matching_dist + 1, CV_32SC1 );
    for( int y = 0; y < increment_indices.rows; y++ ) {
        int * ptr    = increment_indices.ptr<int>(y);
        double v_val = velocity.at<double>(y, 0);
        
        for( int x = 0; x < increment_indices.cols; x++ )
            ptr[x] = static_cast<int>(floor(x * v_val));
    }
    
    
    int y_max = diff_mat.rows;
    
    /* Start trajectory */
    int n_start = N - (matching_dist / 2);
    Mat x( velocity.rows, matching_dist + 1, CV_32SC1 );
    for( int i = 0; i < x.cols; i++ )
        x.col(i) = (n_start + i - 1) * y_max;
    
    
    vector<float> score(diff_mat.rows);
    
    /* Perform the trajectory search to collect the scores */
    for( int s = 0; s < diff_mat.rows; s++ ) {
        Mat y = increment_indices + s;
        Mat( y.size(), y.type(), Scalar(y_max) ).copyTo( y, y > y_max );
        Mat idx_mat = x + y;
        
        Mat temp;
        float min_sum = std::numeric_limits<float>::max();
        for( int row = 0; row < idx_mat.rows; row++ ) {
            float sum = 0.0;
            
            for( int col = 0; col < idx_mat.cols; col++ ){
                int idx = idx_mat.at<int>(row, col);
                sum += diff_mat.at<float>( idx % y_max, idx / y_max );
            }
            min_sum = MIN( min_sum, sum );
        }
        
        score[s] = min_sum;
    }
    
    /* Find the lowest score */
    int min_index = static_cast<int>( std::min_element( score.begin(), score.end() ) - score.begin() );
    double min_val = score[min_index];
    
    /* ... now discard the RWindow region from where we found the lowest score ... */
    for( int i = MAX(0, min_index - RWindow / 2); i < MIN( score.size(), min_index + RWindow / 2); i++ )
        score[i] = std::numeric_limits<double>::max();
    
    /* ... in order to find the second lowest score */
    double min_index_2 = static_cast<int>( std::min_element( score.begin(), score.end() ) - score.begin() );
    double min_val_2 = score[min_index_2];
    
    return pair<int, double> ( min_index + matching_dist / 2, min_val / min_val_2 );
}

/**
 * Return a matching matrix that consists of two rows.
 * First row is the matched image index, second row is the score (the lower the score the better it is)
 */
Mat OpenSeqSLAM::findMatches( Mat& diff_mat, int matching_dist ) {
    int m_dist      = matching_dist + (matching_dist % 2); /* Make sure that distance is even */
    int half_m_dist = m_dist / 2;
    
    /* Match matrix consists of 2 rows, first row is the index of matched image, 
     second is the score. Since the higher score the larger the difference (the weaker the match) 
     we initialize them with maximum value */
    Mat matches( 2, diff_mat.cols, CV_32FC1, Scalar( std::numeric_limits<float>::max() ) );
    
    float * index_ptr = matches.ptr<float>(0);
    float * score_ptr = matches.ptr<float>(1);
    for( int N = half_m_dist + 1; N < (diff_mat.cols - half_m_dist); N++ ) {
        pair<int, double> match = findMatch( diff_mat, N, m_dist );
        
        index_ptr[N] = match.first;
        score_ptr[N] = match.second;
    }
    
    return matches;
}


/**
 * Apply OpenSeqSLAM to sets of image sequences
 *
 * First set is the original sequence of images
 * Second set is the image sequences that we want to match with the 1st set
 * Returns a 2 rows matrix, where first row is the image indices, 
 * and the second row is the score (lower is better)
 */
Mat OpenSeqSLAM::apply( vector<Mat>& set_1, vector<Mat>& set_2 ) {
    Mat diff_mat = calcDifferenceMatrix( set_1, set_2 );
    
    /* Includes additional row on diff matrix with infinite values, to penalize out of bounds cases */
    Mat inf( 1, diff_mat.cols, diff_mat.type(), Scalar(std::numeric_limits<float>::max()) );
    vconcat( diff_mat, inf, diff_mat );
    
    Mat enhanced = enhanceLocalContrast( diff_mat );
    return findMatches( enhanced );
}