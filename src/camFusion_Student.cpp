
#include <iostream>
#include <algorithm>
#include <numeric>
#include <map>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> candidateMatches;
    double count, xsum, xsumsq, ysum, ysumsq;
    for(auto& match: kptMatches)
        // if the match is actually inside our box
        if(boundingBox.keypointInds.count(match.trainIdx)){
            candidateMatches.push_back(match);
            auto prevKptLoc = kptsPrev[match.queryIdx].pt;
            auto currKptLoc = kptsCurr[match.trainIdx].pt;
            count ++;
            xsum += prevKptLoc.x - currKptLoc.x;
            xsumsq += pow(prevKptLoc.x - currKptLoc.x, 2);
            ysum += prevKptLoc.y - currKptLoc.y;
            ysumsq += pow(prevKptLoc.y - currKptLoc.y, 2);
        }
    if(count <0.5)
        return;
    double xmean = xsum/count;
    double ymean = ysum/count;
    double xstd = sqrt( xsumsq/count - xmean*xmean);
    double ystd = sqrt( ysumsq/count - ymean*ymean);
    double std_thresh = 1.0;

    for(auto& match: candidateMatches){
        auto prevKptLoc = kptsPrev[match.queryIdx].pt;
        auto currKptLoc = kptsCurr[match.trainIdx].pt;
        double xdist = prevKptLoc.x - currKptLoc.x;
        double ydist = prevKptLoc.y - currKptLoc.y;
        if( xdist < xmean + std_thresh*xstd &&
            xdist > xmean - std_thresh*xstd &&
            ydist < ymean + std_thresh*ystd &&
            ydist > ymean - std_thresh*ystd)
                boundingBox.kptMatches.push_back(match);
    }

    cout << "Filtered keypoint matches for box ID " << boundingBox.boxID << " from " << candidateMatches.size()
        << " to " << boundingBox.kptMatches.size() << endl;
}

//simple median function inspired by https://stackoverflow.com/questions/2114797/compute-median-of-values-stored-in-vector-c
double median(std::vector<double> x){
    const auto middleItr = x.begin() + x.size()/2;
    std::nth_element(x.begin(), middleItr, x.end());
    return *middleItr;
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC)
{
    double minDist = 100.0;

    std::vector<double> ratios;
    for(int i=0; i<kptMatches.size(); i++){
        for(int j=i+1; j<kptMatches.size(); j++){
            auto match = kptMatches[i];
            auto match2 = kptMatches[j];
            double d0 =  cv::norm(kptsPrev[match.queryIdx].pt-kptsPrev[match2.queryIdx].pt);
            double d1 =  cv::norm(kptsCurr[match.trainIdx].pt-kptsPrev[match2.trainIdx].pt);
            if (d1 > minDist && d0> minDist)
                ratios.emplace_back(d1/d0);
        }

    }
    double median_ratio = median(ratios);
    double dt = 1/frameRate; // seconds between frames
    TTC = -dt/(1.0 - median_ratio);

}



struct NearnessSorter{
    inline bool operator() (const LidarPoint& p1, const LidarPoint& p2){
        return p1.horizontal_distance() < p2.horizontal_distance();
    }
};


double distanceFromPointCloud(std::vector<LidarPoint> &lidarPoints, int dropNearest, int averageOver){
    double count = 0.0;
    double dist_sum = 0.0;
    for(int i=dropNearest; i<dropNearest+averageOver && i<lidarPoints.size(); i++){
        count ++;
        dist_sum += lidarPoints[i].horizontal_distance();
    }

    if(count < 0.5)
        return -1;
    else
        return dist_sum/count;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), NearnessSorter());
    std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), NearnessSorter());

    double prev_dist = distanceFromPointCloud(lidarPointsPrev, 3, 10);
    double curr_dist = distanceFromPointCloud(lidarPointsCurr, 3, 10);
    if(prev_dist>=0 && curr_dist>=0){
        double frames_to_collision = curr_dist/(prev_dist-curr_dist);
        TTC = frames_to_collision/frameRate;
    } else{
        TTC = -1;
    }
}


void assignKeypointsToBoxes(DataFrame &frame){
    for(auto& box: frame.boundingBoxes)
        for(int i=0; i<frame.keypoints.size(); i++)
            if(box.roi.contains(frame.keypoints[i].pt)){
                box.keypoints.push_back(frame.keypoints[i]);
                box.keypointInds.insert(i);
            }
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // count all pairwise occurrences
    std::map<int, std::map<int, int>> pre_matches;

    for(auto match: matches){
        for(auto & prevBox: prevFrame.boundingBoxes){
            for(auto & currBox: currFrame.boundingBoxes){
                if(prevBox.keypointInds.count(match.queryIdx) >0 &&
                    currBox.keypointInds.count(match.trainIdx) >0 ){
                    // if there is no entry for the current box yet, create it
                    if(pre_matches.find(prevBox.boxID)==pre_matches.end())
                        pre_matches[prevBox.boxID] = std::map<int, int>();
                    if(pre_matches[prevBox.boxID].find(currBox.boxID)==pre_matches[prevBox.boxID].end())
                        pre_matches[prevBox.boxID][currBox.boxID] = 0;
                    pre_matches[prevBox.boxID][currBox.boxID] ++;
                    //cout << "matching boxes " << prevBox.boxID << " " << currBox.boxID << endl;
                }
            }
        }
    }

    std::map<int, std::pair<int, int>> best_scores; // (currId, (prevID, count))
    // now find the best current match for each box from prev frame
    // need to take care that we match each current box at most once
    for(auto const& x: pre_matches){
        auto tmp = x.second;
        if(tmp.size()>0){
            int best_count = -1;
            int bestCurrBox = -1;
            for(auto const& y: tmp) {
                auto cnt = y.second;
                if (cnt > best_count) {
                    best_count = cnt;
                    bestCurrBox = y.first;
                }
            }
            // if we never matched to this current box or the old match was worse, just insert it
            if(best_scores.find(bestCurrBox)==best_scores.end() || best_scores[bestCurrBox].second < best_count)
                best_scores[bestCurrBox] = std::pair<int, int>(x.first, best_count);
            //cout << "matched boxes " << x.first << " and " << bestCurrBox << " count " << best_count << endl;
        }
    }

    cout <<"final matching result: " << endl ;
    // now dump the results into the output, remember best_scores is keyed by currId not prevId
    for(auto const& x: best_scores){
        bbBestMatches[x.second.first] = x.first;
        cout << "matched boxes " << x.second.first << " and " << x.first << endl;
    }
}
