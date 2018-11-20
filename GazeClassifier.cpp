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

#include <SequenceCapture.h>
#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <math.h>
#include <iostream>
#include <thread>
#include <mutex>
#include "svm.h"

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

#define NUM_TARGETS 3
#define TRAINING_STEPS 50

//mutex for the continue character
std::mutex mtx;

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

int mode(int labels[15]){
	int best_label = 0;
	int best_num_label = 0;
	int num_of_label = 0;

	for(int label = 0; label < NUM_TARGETS; label++){
		for(int j = 0; j < 15; j++){
			if(labels[j] == label){
				num_of_label++;
			}
		}
		if(num_of_label > best_num_label){
			best_num_label = num_of_label;
			best_label = label;
		}
		num_of_label = 0;
	}

	return best_label;
}

int main(int argc, char **argv)
{

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

	//  explain usage of the program
	cout << "\n\n#######################################################" << endl;
	cout << "\n\n Enter a \'c\' to continue from one phase to next" << endl;
	cout << "There are (# of targets phases) to record data for each target \n then there is a final phase to train and start testing \n\n" << endl;
	cout << "#######################################################\n\n" << endl;

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

	// our mixture of gaussians classifier
	svm k = svm();
	int labeling_pos = -1;
	int train_counter = 0;

	int sequence_number = 0;
	int points_recorded_for_curr_target = TRAINING_STEPS;
	char cont;

	// start the thread to record the the continue character
	std::thread recording_thread(read_buffer, &cont);

	while (true) // this is not a for loop as we might also be reading from a webcam
	{

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

			//eyeball centers 
			cv::Point3f eyeballCentre1(0, 0, 0);
			cv::Point3f eyeballCentre2(0, 0, 0);

			cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);

			// If tracking succeeded and we have an eye model, estimate gaze
			if (detection_success && face_model.eye_model)
			{
				GazeAnalysis::EstimateGaze(face_model, gazeDirection0, eyeballCentre1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true);
				GazeAnalysis::EstimateGaze(face_model, gazeDirection1, eyeballCentre2, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, false);

				//for testing
				cout << eyeballCentre1 << endl;

				float data_point[6] = {static_cast<float>(pose_estimate[0]), 
									   static_cast<float>(pose_estimate[1]), 
									   static_cast<float>(pose_estimate[2]),
									   gazeDirection0.x, 
									   gazeDirection0.y, 
									   gazeDirection0.z};			

				if(points_recorded_for_curr_target >= TRAINING_STEPS && read_char_async(&cont) == 'c'){
					points_recorded_for_curr_target = 0;
					labeling_pos++;
				}
				if(labeling_pos < NUM_TARGETS && points_recorded_for_curr_target < TRAINING_STEPS){
					k.add(data_point, labeling_pos);
					cout << "Look at position  " << labeling_pos << ": " << points_recorded_for_curr_target << endl;
					points_recorded_for_curr_target++;
				} else if(labeling_pos == (NUM_TARGETS)){
					k.cluster();
					labeling_pos++;
				} else if(labeling_pos > NUM_TARGETS) {
					cout << k.eval(data_point) << endl;
				}
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
				return(0);
			}

			// Grabbing the next frame in the sequence
			rgb_image = sequence_reader.GetNextFrame();

		}

		// Reset the model, for the next video
		face_model.Reset();
		sequence_reader.Close();

		sequence_number++;

	}
	return 0;
}

