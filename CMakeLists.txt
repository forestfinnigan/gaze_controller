# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
# Fabian Timm's pupil finding library
include_directories(${EyeLike_SOURCE_DIR}/include)
	
add_executable(GazeClassifier GazeClassifier.cpp)
target_link_libraries(GazeClassifier LandmarkDetector)
target_link_libraries(GazeClassifier FaceAnalyser)
target_link_libraries(GazeClassifier GazeAnalyser)
target_link_libraries(GazeClassifier Utilities)
target_link_libraries(GazeClassifier EyeLike)

install (TARGETS GazeClassifier DESTINATION bin)