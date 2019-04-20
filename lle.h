#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include "lwlr.h"

KalmanFilter setup_KF(float x, float y){
    // From http://opencvexamples.blogspot.com/2014/01/kalman-filter-implementation-tracking.html
    KalmanFilter KF(2, 2, 0);
   
    // intialization of KF...

    KF.transitionMatrix = (Mat_<float>(2, 2) << 1,0,   0,1);

    KF.statePre.at<float>(0) = x;
    KF.statePre.at<float>(1) = y;
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(.5));
    setIdentity(KF.measurementNoiseCov, Scalar::all(10));
    setIdentity(KF.errorCovPost, Scalar::all(.1));

    return KF; 
}


class lle
{
    static const int DATA_DIM = 9; 
    float x_raw = 0.0;
    float y_raw = 0.0; 
    KalmanFilter KFl;
    KalmanFilter KFr; 
    int KF_initl = 0; 
    int KF_initr = 0; 
    lwlr l = lwlr(4);
    lwlr r = lwlr(4); 
    cv::Mat A = Mat::zeros(3,3, CV_32FC1);
    cv::Mat b = Mat::zeros(3,1, CV_32FC1);
    cv::Mat c = Mat::zeros(3,1, CV_32FC1);
    Mat out_image =  Mat::zeros( 500, 500, CV_8UC3 );

    public: 
    lle() 
    { 

    } 
    void load(string sl, string sr) 
    {
        float data[DATA_DIM];
        float x,z,d;
        char c;
        string str; 
        //Stack overflow
        std::ifstream infile_l(sl);
        if (infile_l.is_open()) {
            while (infile_l >> x >> c >> z && c == ','){
                for (int i = 0; i < DATA_DIM; i++) {
                    infile_l >> c >> d;
                    data[i] = d;  
                }
                add(data, x, z, 1, NULL); 
            }
        }
        infile_l.close();

        std::ifstream infile_r(sr);
        if (infile_r.is_open()) {
            while (infile_r >> x >> c >> z && c == ',') {
                for (int i = 0; i < DATA_DIM; i++) {
                    infile_r >> c >> d;
                    data[i] = d;
                }
                add(data, x, z, 0, NULL);
            }
        }
        infile_r.close(); 
    }
    void add(float d_model[DATA_DIM], float x, float z, int is_l, ofstream *txtOut){
        if (txtOut != NULL)
            (*txtOut) << x << "," << z << "," << d_model[0] << "," << d_model[1] << "," << d_model[2] << "," << 
            d_model[3] << "," << d_model[4] << "," << d_model[5] << "," << 
            d_model[6] << "," << d_model[7] << "," << d_model[8] << endl;
        // cout << x << "," << z << "," << d_model[0] << "," << d_model[1] << "," << d_model[2] << "," << 
        //     d_model[3] << "," << d_model[4] << "," << d_model[5] << "," << 
        //     d_model[6] << "," << d_model[7] << "," << d_model[8] << endl;
        if (is_l == 1) {
            l.add(d_model, x, 0, z);
        }
        else
            r.add(d_model, x, 0, z);
    }
    void project(cv::Mat &out_x, cv::Mat &P, cv::Point3f e, cv::Point3f p, cv::Vec6d pe, int &KF_init, KalmanFilter &KF, int is_l, ofstream *txtOut){
        float eye[9] = {e.x, e.y, e.z, p.x, p.y, p.z, (float)pe[3], (float)pe[4], (float)pe[5]};
        float x,y,z; 
        if (is_l == 1) {
            l.eval(x, y, z, eye);
        }
        else
            r.eval(x, y, z, eye);  

        cv::Mat raw = Mat::zeros(3,1,CV_32FC1);
        raw.at<float>(0,0) = x;
        raw.at<float>(1,0) = y;
        raw.at<float>(2,0) = z; 

        //Initialize the filter if needed
        if (KF_init == 0){
            KF = setup_KF(raw.at<float>(0,0), raw.at<float>(2,0));
            KF_init = 1;
        }

        // //Step the kalman filter 
        // //From http://opencvexamples.blogspot.com/2014/01/kalman-filter-implementation-tracking.html
        Mat prediction = KF.predict(); 
         Mat_<float> measurement(2,1); measurement.setTo(Scalar(0));
        measurement(0) = raw.at<float>(0,0);
        measurement(1) = raw.at<float>(2,0);
        KF.correct(measurement).copyTo(out_x);
        KF.errorCovPost.copyTo(P);
    }
      
    int eval(float &x, float &y, float &z, float &x_raw, float &y_raw, float &z_raw, cv::Point3f el, cv::Point3f pl, cv::Point3f er, cv::Point3f pr, cv::Vec6d pe, ofstream *txtOut){
        cv::Mat xl = Mat::zeros(2,1, CV_32FC1);
        cv::Mat xr = Mat::zeros(2,1, CV_32FC1);
        cv::Mat Pl = Mat::zeros(2,2, CV_32FC1);
        cv::Mat Pr = Mat::zeros(2,2, CV_32FC1);
        project(xl,Pl,el,pl,pe,KF_initl, KFl, 1, txtOut);
        project(xr,Pr,er,pr,pe,KF_initr, KFr, 0, txtOut);

 
        cv::Mat point = (Pl.inv() + Pr.inv()).inv() * (Pl.inv() * xl + Pr.inv() * xr);
        x = point.at<float>(0,0);
        z = point.at<float>(1,0); 
        if (txtOut != NULL)
            (*txtOut) << x << "," << z << "," << el.x << "," << el.y << "," << el.z << ","
            << pl.x << "," << pl.y << "," << pl.z << ","
            << er.x << "," << er.y << "," << er.z << ","
            << pr.x << "," << pr.y << "," << pr.z << ","
            << pe[3] << "," << pe[4] << "," << pe[5] << endl; 
        return 1;
    }
};