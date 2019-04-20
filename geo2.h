#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/video/tracking.hpp"

#define DATA_DIM 6

using namespace cv;

KalmanFilter setup_KF(float x, float y){
    // From http://opencvexamples.blogspot.com/2014/01/kalman-filter-implementation-tracking.html
    KalmanFilter KF(4, 2, 0);
   
    // intialization of KF...

    KF.transitionMatrix = (Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);

    KF.statePre.at<float>(0) = x;
    KF.statePre.at<float>(1) = y;
    KF.statePre.at<float>(2) = 0;
    KF.statePre.at<float>(3) = 0;
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
    setIdentity(KF.measurementNoiseCov, Scalar::all(10));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    return KF; 
}


class geo2
{
    float gamma;
    float x_raw = 0.0;
    float y_raw = 0.0; 
    KalmanFilter KF; 
    int KF_init = 0; 
    cv::Mat A = Mat::zeros(3,3, CV_32FC1);
    cv::Mat b = Mat::zeros(3,1, CV_32FC1);
    cv::Mat c = Mat::zeros(3,1, CV_32FC1);
    float d; 


    public: 
    geo2(float gamma, cv::Mat rot, cv::Mat trans) 
    { 
        this->gamma = gamma; 
        rot.col(0).copyTo(A.col(0));
        rot.col(1).copyTo(A.col(1));
        trans.col(0).copyTo(c.col(0));

    } 
    void add(float data_point[DATA_DIM], float x, float y){

    }
    void project(float &x, float &y, float &x_raw, float &y_raw, float d[DATA_DIM]){
        A.at<float>(0,2) = d[0];
        A.at<float>(1,2) = d[1];
        A.at<float>(2,2) = d[2]; 
        b.at<float>(0,0) = d[3] - c.at<float>(0,0);
        b.at<float>(1,0) = d[4] - c.at<float>(1,0);
        b.at<float>(2,0) = d[5] - c.at<float>(2,0); 

        cv::Mat raw = A.inv() * b;

        x_raw = raw.at<float>(0,0);
        y_raw = raw.at<float>(1,0);

        //Initialize the filter if needed
        if (KF_init == 0){
            KF = setup_KF(x_raw, y_raw);
            KF_init = 1;
        }

        //Step the kalman filter 
        //From http://opencvexamples.blogspot.com/2014/01/kalman-filter-implementation-tracking.html
        Mat prediction = KF.predict();
        Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));
        measurement(0) = x_raw;
        measurement(1) = y_raw; 
        Mat estimated = KF.correct(measurement);

        x = estimated.at<float>(0);
        y = estimated.at<float>(1);
    }
      
    int eval(float &x, float &y, float &x_raw, float &y_raw, float d[DATA_DIM])
    {
        project(x,y,x_raw,y_raw,d);
        return 1;
    }
};