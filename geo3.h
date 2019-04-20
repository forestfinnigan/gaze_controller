#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/video/tracking.hpp"

#define DATA_DIM 6

using namespace cv;

KalmanFilter setup_KF(float x, float y, float z){
    // From http://opencvexamples.blogspot.com/2014/01/kalman-filter-implementation-tracking.html
    KalmanFilter KF(3, 3, 0);
   
    // intialization of KF...

    KF.transitionMatrix = (Mat_<float>(3, 3) << 1,0,0,   0,1,0,  0,0,1);

    KF.statePre.at<float>(0) = x;
    KF.statePre.at<float>(1) = y;
    KF.statePre.at<float>(2) = z;
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(.5));
    setIdentity(KF.measurementNoiseCov, Scalar::all(10));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    return KF; 
}


class geo3
{
    KalmanFilter KF; 
    int KF_init = 0; 
    cv::Mat A = Mat::zeros(3,2, CV_32FC1);
    cv::Mat b = Mat::zeros(3,1, CV_32FC1);
    cv::Mat rot_inv;
    cv::Mat trans; 

    public: 
    geo3(cv::Mat rot, cv::Mat trans, cv::Mat shift) 
    {  
        this->rot_inv = rot.inv();
        this->trans = rot * (shift) + trans;;
    } 
    void add(float d_model[DATA_DIM], float x, float z, int is_l, ofstream *txtOut){

    }
    void project(float &x, float &y, float &z, float &x_raw, float &y_raw, float &z_raw, cv::Point3f el, cv::Point3f pl, cv::Point3f er, cv::Point3f pr){
        A.at<float>(0,0) = el.x;
        A.at<float>(1,0) = el.y;
        A.at<float>(2,0) = el.z;
        A.at<float>(0,1) = -er.x;
        A.at<float>(1,1) = -er.y;
        A.at<float>(2,1) = -er.z; 
        b.at<float>(0,0) = pr.x - pl.x;
        b.at<float>(1,0) = pr.y - pl.y;
        b.at<float>(2,0) = pr.z - pl.z;

        cv::Mat gammas = (A.t() * A).inv() * A.t() * b;

        float gamma = gammas.at<float>(0,0);

        cv::Point3f raw = gamma * el + pl;

        //Initialize the filter if needed
        if (KF_init == 0){
            KF = setup_KF(raw.x, raw.y, raw.z);
            KF_init = 1;
        }

        //Step the kalman filter 
        //From http://opencvexamples.blogspot.com/2014/01/kalman-filter-implementation-tracking.html
        Mat prediction = KF.predict();
        Mat_<float> measurement(3,1); measurement.setTo(Scalar(0));
        measurement(0) = raw.x;
        measurement(1) = raw.y;
        measurement(2) = raw.z; 
        Mat estimated = KF.correct(measurement);

        x = estimated.at<float>(0);
        y = estimated.at<float>(1);
        z = estimated.at<float>(2);
        x_raw = raw.x;
        y_raw = raw.y;
        z_raw = raw.z; 
    }
      
    int eval(float &x, float &y, float &z, float &x_raw, float &y_raw, float &z_raw, cv::Point3f el, cv::Point3f pl, cv::Point3f er, cv::Point3f pr, ofstream *txtOut)
    {
        if (txtOut != NULL)
            (*txtOut) << el.x << "," << el.y << "," << el.z << ","
            << pl.x << "," << pl.y << "," << pl.z << ","
            << er.x << "," << er.y << "," << er.z << ","
            << pr.x << "," << pr.y << "," << pr.z << endl;
        project(x,y,z, x_raw, y_raw, z_raw, el,pl,er,pr);

        cv::Mat pointMat = (Mat_<float>(3, 1) << x, y, z);

        pointMat = this->rot_inv * (pointMat - this->trans);

        x = pointMat.at<float>(0,0);
        y = pointMat.at<float>(1,0); 
        z = pointMat.at<float>(2,0);        

        return 1;
    }
};