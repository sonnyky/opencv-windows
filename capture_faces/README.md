# capture_faces
This is a utility tool to capture images of faces and stores them into separate folders according to the name input

# usage
From the command line, run the exe with the following parameters, in the following order
* path to the settings file (XML) that specifies capture parameters
* name of the user, used to create separate folders in the data folder

### example
capture_faces.exe C\:Desktop\Settings\setting.xml tom
This will create a folder in the specified data folder path called __tom__ with the captured face images numbered from 0 to whatever number.
