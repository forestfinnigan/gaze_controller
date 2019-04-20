///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////
// FaceTrackingVid.cpp : Defines the entry point for the console application for tracking faces in videos.

// Libraries for landmark detection (includes CLNF and CLM modules)
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"
#include <iostream>
#include <fstream>

#include <SequenceCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <math.h>
#include <iostream>
#include <thread>
#include <mutex>
#include "PlanarVisualization.h"
// #include "svm.h"
//#include "geo2prob.h"
#include "geo3.h"
#include "GazeCamera.h"
//#include "lle.h"
#include <time.h>
#include <sstream>
#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort(const std::string & error)
{
	std::cout << error << std::endl;
	abort();
}

void read_buffer(char * buffer){
	for(;;){
		cin >> *buffer;
	}
}

char read_char_async(char * buffer){
	char ret;
	ret = *buffer;
	*buffer = '\0';
	return ret;
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

//To make the calculations
// geo g =  geo(.5);
// lwlr g =  lwlr(5);
Mat rot =  (Mat_<float>(3,3) << .99, -.008, -.009, -.012, -.93, -.34, -.0061, -.35, -.93);
Mat trans =  (Mat_<float>(3,1) << -100,200,530);
Mat shift =  (Mat_<float>(3,1) << 120,-120,250);
// cout << rot << endl;
// cout << trans << endl;
geo3 g = geo3(rot, trans, shift);
//lle g = lle();
//geo2prob g = geo2prob(rot, trans, shift);
// g.load()
// geo2prob g_no_l = geo2prob(rot, trans, shift);
//svm s = svm(); 
PlanarVisualization pv(920, 500, 200); 

//To read in clicks
// float data_point[6]; 
void cbmouse(int event, int x, int y, int flags, void*userdata)
{
   	if (event == EVENT_LBUTTONDOWN)
   	{
   		x = x - 460;
   		y = 500 - y;
   		cout << x << "," << y << endl;
      	// g.add(data_point, x, y);
      	//g.add((float *)userdata, x, y); 
   	}
}

int main(int argc, char **argv)
{

	//
	vector<string> arguments = get_arguments(argc, argv);

	// no arguments
	if(arguments.size() == 1){
		// allowing the user to enter the arguement in the terminal
		arguments.push_back("-device");
		string dev_number;
		cout << "Enter the camera device number: ";
		cin >> dev_number;
		cin.ignore();
		cout << endl;
		arguments.push_back(dev_number);
	}

	LandmarkDetector::FaceModelParameters det_parameters(arguments);

	// The modules that are being used for tracking
	LandmarkDetector::CLNF face_model(det_parameters.model_location);
	if (!face_model.loaded_successfully)
	{
		cout << "ERROR: Could not load the landmark detector" << endl;
		return 1;
	}

	if (!face_model.eye_model)
	{
		cout << "WARNING: no eye model found" << endl;
	}

	// Open a sequence
	Utilities::SequenceCapture sequence_reader;

	// A utility for visualizing the results (show just the tracks)
	Utilities::Visualizer visualizer(true, false, false, false);

	// Tracking FPS for visualization
	Utilities::FpsTracker fps_tracker;
	fps_tracker.AddFrame();

	//To display the output
	int h = 500;
	int w = 920;  
	float data_l[9];
	float data_r[9];
	char position;
	Mat out_image =  Mat::zeros( h, w, CV_8UC3 );
	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	setMouseCallback("Display window", cbmouse, data_l);
	imshow("Display window", out_image); 
	//waitKey(0);

	while (true) // this is not a for loop as we might also be reading from a webcam
	{
		// time_t t = time(NULL);
		// ofstream txtOut;
		// ofstream txtOutl;
		// ofstream txtOutr;
		// struct tm tm = *localtime(&t);
		// ostringstream os;
		// ostringstream osl;
		// ostringstream osr; 
		// os << "data/data_" << tm.tm_mon + 1 << "-" << tm.tm_mday << "_" << tm.tm_hour << ":" << tm.tm_min << ".txt";
		// osl << "data/data_l" << tm.tm_mon + 1 << "-" << tm.tm_mday << "_" << tm.tm_hour << ":" << tm.tm_min << ".txt";
		// osr << "data/data_r" << tm.tm_mon + 1 << "-" << tm.tm_mday << "_" << tm.tm_hour << ":" << tm.tm_min << ".txt";
		// string s = os.str();
		// string sl = osl.str();
		// string sr = osr.str();
		// txtOut.open (s);
		// txtOutl.open (sl);
		// txtOutr.open (sr);
		// The sequence reader chooses what to open based on command line arguments provided
		if (!sequence_reader.Open(arguments))
			break;

		INFO_STREAM("Device or file opened");

		cv::Mat rgb_image = sequence_reader.GetNextFrame();

		INFO_STREAM("Starting tracking");
		while (!rgb_image.empty()) // this is not a for loop as we might also be reading from a webcam
		{
			// Reading the images
			cv::Mat_<uchar> grayscale_image = sequence_reader.GetGrayFrame();

			// The actual facial landmark detection / tracking
			bool detection_success = LandmarkDetector::DetectLandmarksInVideo(rgb_image, face_model, det_parameters, grayscale_image);

			// Gaze tracking, absolute gaze direction
			cv::Point3f gazeDirection0(0, 0, -1);
			cv::Point3f gazeDirection1(0, 0, -1);
			cv::Point3f gazePos0(0, 0, -1);
			cv::Point3f gazePos1(0, 0, -1);


			cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);

			// If tracking succeeded and we have an eye model, estimate gaze
			if (detection_success && face_model.eye_model)
			{
				GazeAnalysis::EstimateGaze(face_model, gazeDirection0, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, false, gazePos0);
				GazeAnalysis::EstimateGaze(face_model, gazeDirection1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true, gazePos1);

			
				float x_curr = 0.0;
				float y_curr = 0.0;
				float z_curr = 0.0;
				float x_raw = 0.0;
				float y_raw = 0.0;
				float z_raw = 0.0;
				data_l[0] = gazeDirection0.x;
				data_l[1] = gazeDirection0.y;
				data_l[2] = gazeDirection0.z;
				data_l[3] = gazePos0.x;
				data_l[4] = gazePos0.y;
				data_l[5] = gazePos0.z;
				data_l[6] = pose_estimate[3];
				data_l[7] = pose_estimate[4];
				data_l[8] = pose_estimate[5];
				data_r[0] = gazeDirection1.x;
				data_r[1] = gazeDirection1.y;
				data_r[2] = gazeDirection1.z;
				data_r[3] = gazePos1.x;
				data_r[4] = gazePos1.y;
				data_r[5] = gazePos1.z;
				data_r[6] = pose_estimate[3];
				data_r[7] = pose_estimate[4];
				data_r[8] = pose_estimate[5];

				//cout << pose_estimate[3] << " " << pose_estimate[4] << " " << pose_estimate[5] << endl; 
				//cout << gazePos0 << endl;

				//int valid = g.eval(x_curr, y_curr, z_curr, x_raw, y_raw, z_raw, gazeDirection0, gazePos0, gazeDirection1, gazePos1, pose_estimate, NULL);
				//int valid_no_l = g_no_l.eval(x_raw, y_raw, z_raw, x_curr, y_curr, z_curr, gazeDirection0, gazePos0, gazeDirection1, gazePos1);
				int valid = g.eval(x_curr, y_curr, z_curr, x_raw, y_raw, z_raw, gazeDirection0, gazePos0, gazeDirection1, gazePos1, NULL);
				
				
				//show the values
				out_image =  Mat::zeros( h, w, CV_8UC3 );

				// cout << x_curr << " " << y_curr << endl; 
				// cout << z_curr << endl;

				// if (valid == 1) {
				// 	txtOut << data_l[0] << ", " << data_l[1] << ", " << data_l[2] << ", " << data_l[3] << ", " << data_l[4] << ", " << data_l[5] << ", ";
				// 	txtOut << data_r[0] << ", " << data_r[1] << ", " << data_r[2] << ", " << data_r[3] << ", " << data_r[4] << ", " << data_r[5] << endl;
				// }

				int color = z_curr / 2 + 125;
				if (color > 255)
					color = 255;
				if (color < 0)
					color = 0; 
				ellipse( out_image,
           			Point(460 - x_curr, 500-z_curr),
           			Size( w/80.0, w/80.0 ),
           			0,
           			0,
           			360,
           			Scalar( color, 255, 255),
           			2,
           			8);
				// ellipse( out_image,
    //         			Point(460 - x_raw, 500-z_raw),
    //         			Size( w/40.0, w/40.0 ),
    //         			0,
    //         			0,
    //         			360,
    //         			Scalar( 255, 0, 255),
    //         			2,
    //         			8);
				pv.drawGrid(out_image);

				imshow( "Display window", out_image);
			}

			// Keeping track of FPS
			fps_tracker.AddFrame();
	
			// Displaying the tracking visualizations
			visualizer.SetImage(rgb_image, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);
			visualizer.SetObservationLandmarks(face_model.detected_landmarks, face_model.detection_certainty, face_model.GetVisibilities());
			visualizer.SetObservationPose(pose_estimate, face_model.detection_certainty);
			visualizer.SetObservationGaze(gazeDirection0, gazeDirection1, LandmarkDetector::CalculateAllEyeLandmarks(face_model), LandmarkDetector::Calculate3DEyeLandmarks(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy), face_model.detection_certainty);
			visualizer.SetFps(fps_tracker.GetFPS());
			// detect key presses (due to pecularities of OpenCV, you can get it when displaying images)
			char character_press = visualizer.ShowObservation();


			// restart the tracker
			if (character_press == 'r')
			{
				face_model.Reset();
			}
			// quit the application
			else if (character_press == 'q')
			{
				// txtOut.close();
				// txtOutl.close();
				// txtOutr.close();
				return(0);
			}
			else if (character_press >= '0' && character_press <= '9'){
				position = character_press;
				float x;
				float y;
				pv.getPoint(x, y, position - '0');
				x = 460 - x;
   				y = 500 - y;
				cout << x << " " << y << endl; 
				g.add(data_l, x, y, 1, NULL);
				g.add(data_r, x, y, 0, NULL);

			}
			else{
				position = 'a';
			}

			// Grabbing the next frame in the sequence
			rgb_image = sequence_reader.GetNextFrame();

		}

		// Reset the model, for the next video
		face_model.Reset();
		sequence_reader.Close();

	}
	return 0;
}

