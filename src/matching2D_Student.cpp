#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef,
        cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    double t = (double)cv::getTickCount();

    if (matcherType.compare("MAT_BF") == 0)
    {

        int normType;
        if(descriptorType.compare("DES_BINARY")==0)
            normType= cv::NORM_HAMMING;
        else
            normType = cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // taken from https://answers.opencv.org/question/59996/flann-error-in-opencv-3/
        matcher = cv::makePtr<cv::FlannBasedMatcher>(
                cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        // do the distance ratio test
        for(auto& vec: knn_matches)
            if(vec.size()==2){
                if(10*vec[0].distance <  8*vec[1].distance)
                    matches.push_back(vec[0]);
                else if(10*vec[1].distance <  8*vec[0].distance)
                    matches.push_back(vec[1]);
            }else if(vec.size()==1) {
                matches.push_back(vec[0]);
            }else{
                //cout << "malformed vector!" << endl;
            }
    }

    cout << "*** Matching using " << matcherType << " got " << matches.size() << " matches" << endl;
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " Matching done in " << 1000 * t / 1.0 << " ms" << endl;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)//BRIEF, ORB, FREAK, AKAZE, SIFT
    {
        int bits = 32;
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bits);//...
    }else if (descriptorType.compare("ORB") == 0)//BRIEF, ORB, FREAK, AKAZE, SIFT
    {
        extractor = cv::ORB::create();//...
    }else if (descriptorType.compare("FREAK") == 0)//BRIEF, ORB, FREAK, AKAZE, SIFT
    {
        extractor = cv::xfeatures2d::FREAK::create();//...
    }else if (descriptorType.compare("AKAZE") == 0)//BRIEF, ORB, FREAK, AKAZE, SIFT
    {
        extractor = cv::AKAZE::create();//...
        for(auto & kp: keypoints)
            kp.class_id = 0; // A magic fix without which AKAZE will fail
    }else if (descriptorType.compare("SIFT") == 0)//BRIEF, ORB, FREAK, AKAZE, SIFT
    {
        extractor = cv::xfeatures2d::SIFT::create();//...
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    return 1000.0*t;
}

void visualize(cv::Mat& img, vector<cv::KeyPoint> &keypoints)
{
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "Shi-Tomasi Corner Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasiHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, bool useHarris)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarris, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    if(useHarris) {
        cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms"
             << endl;
    }else{
        cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms"
             << endl;
    }
    // visualize results
    if (bVis)
        visualize(img, keypoints);

    return 1000.0*t;
}


double detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis){
    if(detectorType.compare("SHITOMASI") == 0)
        return detKeypointsShiTomasiHarris(keypoints, img, bVis, false);

    if(detectorType.compare("HARRIS") == 0)
        return detKeypointsShiTomasiHarris(keypoints, img, bVis, true);

    double t = (double)cv::getTickCount();

    //FAST, BRISK, ORB, AKAZE, SIFT
    if(detectorType.compare("FAST") == 0){
        int threshold = 10;
        FAST(img, keypoints, threshold);
    }else{
        cv::Ptr<cv::FeatureDetector> detector;
        if(detectorType.compare("BRISK") == 0){
            detector = cv::BRISK::create();
        }else if(detectorType.compare("ORB") == 0){
            detector = cv::ORB::create();
        }else if(detectorType.compare("AKAZE") == 0){
            detector = cv::AKAZE::create();
        }else if(detectorType.compare("SIFT") == 0){
            detector = cv::xfeatures2d::SIFT::create();
        }
        detector->detect(img, keypoints);
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms"
         << endl;

    // visualize results
    // visualize results
    if (bVis)
        visualize(img, keypoints);
    return 1000.0*t;
}