#include "opencv2/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>

using namespace cv;

using namespace std; 

class lwlr
{
    float gamma;
    static const int DATA_DIM = 2;
    int k;  
    float n_data;
    cv::Mat labels, data;
    cv::Mat m, s; 
    cv::Mat std;
    cv::Mat old_m, old_s; 
    void run_stats() {
        if (n_data > 1) {
            std = Mat::zeros(1, DATA_DIM, CV_32FC1);
            for (int i = 0; i < DATA_DIM; i++) {
                std.at<float>(i) = sqrt(s.at<float>(0,i) / (n_data - 1)); 
            }
            for (int i = 0; i < n_data; i++) {
                data.row(i) = (data.row(i) - m) / std; 
            }
        }
    }
    float dist(cv::Mat r1, cv::Mat r2){
        return (r1 - r2).dot(r1 - r2); 
    }
    void farthest(int *n_indices, cv::Mat point, int &index, float &max){ 
        max = 0; 
        for (int i = 0; i < k; i++){
            float d = dist(data.row(n_indices[i]), point);
            if (d > max){
                index = i; 
                max = d; 
            }
        }
    }
    void k_closest(cv::Mat &neighbors, cv::Mat &neighbor_values, cv::Mat point, int k){
        int index = 0;
        float max = 0;
        int * n_indices;
        n_indices = new int [k]();
        for (int i = 0; i < k ; i++) 
            n_indices[i] = i; 

        farthest(n_indices, point, index, max);
        for(int i = k; i < n_data; i++){
            if (dist(point, data.row(i)) < max){
                n_indices[index] = i;
                farthest(n_indices, point, index, max);
            }
        }
        for(int i = 0; i < k; i++){
            neighbors.push_back(data.row(n_indices[i]));
            neighbor_values.push_back(labels.row(n_indices[i]));
        }
        delete[] n_indices;
    }
    void get_interp(cv::Mat &point, cv::Mat &interp){
        cv::Mat A = Mat::zeros( 0, DATA_DIM, CV_32FC1);
        cv::Mat b = Mat::zeros( 0, 3, CV_32FC1);
        k_closest(A, b, point, k);

        //cout << b << endl;
 
        cv::Mat id = Mat::eye(k, k, CV_32FC1);
      
        cv::Mat C = A * A.t() + .1 * id;
    
        cv::Mat C_inv = C.inv();
        float alpha = 1;
        float beta = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                beta = beta + C_inv.at<float>(i,j);
                alpha = alpha - C_inv.at<float>(i,j) * point.dot(A.row(j));
            }
        }
        float lambda = alpha / beta; 
        for (int i = 0; i < k; i++) {
            float w = 0; 
            for (int j = 0; j < k; j++) {
                w = w + C_inv.at<float>(i,j) * (point.dot(A.row(j)) + lambda); 
            }
            interp = interp + w * b.row(i);
        }


    }
    public: 
    lwlr(int k) { 
        n_data = 0; 
        this->k = k; 
    } 
    void add(float data_point[DATA_DIM], float x, float y, float z){
        n_data++; 
        cv::Mat data_point_mat(1,DATA_DIM, CV_32FC1, data_point);  
        float label [3] = {x, y, z}; 
        Mat labelMat(1, 3, CV_32FC1, label);

        if (n_data == 1){
            data_point_mat.copyTo(m);
            data_point_mat.copyTo(old_m);
            old_s = Mat::zeros(1, DATA_DIM, CV_32FC1);
        }

        m = old_m + (data_point_mat - old_m) / n_data; 
        s = old_s + (data_point_mat - old_m).mul(data_point_mat - m); 

        m.copyTo(old_m);
        s.copyTo(old_s);

        if (n_data > 1) {
            std = Mat::ones(1, DATA_DIM, CV_32FC1);
            for (int i = 0; i < DATA_DIM; i++) {
                std.at<float>(i) = sqrt(s.at<float>(0,i) / (n_data - 1)); 
            }

            //data.push_back((data_point_mat - m) / std);
            data.push_back(data_point_mat);
            labels.push_back(labelMat);
        } else {
            data.push_back(data_point_mat);
            labels.push_back(labelMat);
        }

        // cout << data << endl;

    }
      
    int eval(float &x, float &y, float &z, float d[DATA_DIM])
    {
        if (n_data > k){
            cv::Mat interp = Mat::zeros( 1, 3, CV_32FC1);
            cv::Mat point(1,DATA_DIM, CV_32FC1, d);
            // cout << point << endl; 
            //point = (point - m) / std; 
            //cout << point << endl; 
            get_interp(point, interp);
            x = interp.at<float>(0,0);
            y = interp.at<float>(0,1);
            z = interp.at<float>(0,2);
            return 1;
        }
        return 0; 

    }
};