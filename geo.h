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


class geo
{
    float gamma;
    float x_raw = 0.0;
    float y_raw = 0.0; 
    KalmanFilter KF; 
    int num_data = 0; 
    int KF_init = 0;
    cv::Mat labels, data;
    cv::Mat H = Mat::zeros(3,3, CV_32FC1);


    public: 
    geo(float gamma) 
    { 
        this->gamma = gamma; 
    } 
    void add(float data_point[DATA_DIM], float x, float y){
        float x_data;
        float y_data;
        project(x_data,y_data,x_raw,y_raw,data_point);
        // cv::Mat data_point_mat1(1,8, CV_32FC1, {static_cast<float>(x_data), static_cast<float>(y_data),
        //                                         static_cast<float>(1), static_cast<float>(0),
        //                                         static_cast<float>(0), static_cast<float>(0),
        //                                         static_cast<float>(-x * x_data), static_cast<float>(-x * y_data)});  
        float data_point1[8] = {static_cast<float>(x_data), static_cast<float>(y_data),
                                static_cast<float>(1), static_cast<float>(0),
                                static_cast<float>(0), static_cast<float>(0),
                                static_cast<float>(-x * x_data), static_cast<float>(-x * y_data)};
        cv::Mat data_point_mat1(1,8, CV_32FC1,data_point1);
        data.push_back(data_point_mat1);
        float data_point2[8] = {static_cast<float>(0), static_cast<float>(0),
                                static_cast<float>(0), static_cast<float>(x_data),
                                static_cast<float>(y_data), static_cast<float>(1),
                                static_cast<float>(-y * x_data), static_cast<float>(-y * y_data)};
        cv::Mat data_point_mat2(1,8, CV_32FC1, data_point2);  
        data.push_back(data_point_mat2);
        Mat labelMat1(1, 1, CV_32FC1, {static_cast<float>(x)});
        labels.push_back(labelMat1);
        Mat labelMat2(1, 1, CV_32FC1, {static_cast<float>(y)});
        labels.push_back(labelMat2);

        num_data++; 

        if (num_data > 3){
            Mat h = (data.t() * data).inv() * data.t() * labels;
            cout << h << endl;
            H.at<float>(0,0) = h.at<float>(0);
            H.at<float>(0,1) = h.at<float>(1);
            H.at<float>(0,2) = h.at<float>(2);
            H.at<float>(1,0) = h.at<float>(3);
            H.at<float>(1,1) = h.at<float>(4);
            H.at<float>(1,2) = h.at<float>(5);
            H.at<float>(2,0) = h.at<float>(6);
            H.at<float>(2,1) = h.at<float>(7);
            H.at<float>(2,2) = 1;
            cout << H << endl;

        }

    }
    void homography(float &x, float &y){
        if (num_data > 3){
            Mat U(3, 1, CV_32FC1, {1});
            U.at<float>(0) = x;
            U.at<float>(1) = y;
            Mat V = H * U;
            if (V.at<float>(2) != 0){
                //cout << "before" <<  x << "," << y << endl;
                x = V.at<float>(0) / V.at<float>(2);
                y = V.at<float>(1) / V.at<float>(2);
                //cout << "after" <<  x << "," << y << endl; 
            }
        }
    }
    void project(float &x, float &y, float &x_raw, float &y_raw, float d[DATA_DIM]){
        x_raw = (d[3] - d[0] / d[2] * d[5]);
        y_raw = (d[4] - d[1] / d[2] * d[5]); 

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


        homography(x, y);
    }
      
    int eval(float &x, float &y, float &x_raw, float &y_raw, float d[DATA_DIM])
    {
        int valid = 0; 
        project(x,y,x_raw,y_raw,d);
        if (num_data > 3)
            valid = 1; 
        return 1;
        
    }
};