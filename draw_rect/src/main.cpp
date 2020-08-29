#include <main.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	if (argc == 3) {
		cout << "Please specify input files and image." << endl;
		exit(1);
	}

	char* imageName = argv[1];

	char* jsonName = argv[2];

	char* jsonFacesName = argv[3];

	ifstream personRect(jsonName, ifstream::binary);
	Json::Value person;
	personRect >> person;

	ifstream personRects(jsonFacesName, ifstream::binary);
	Json::Value rects;
	personRects >> rects;

	Mat image;
	image = imread(imageName, IMREAD_COLOR);

	for (int i = 0; i < person.size(); i++) {
		if (person[i]["candidates"].size() != 0) {
			// Check if the person matches the detected faces
			for (int j = 0; j < rects.size(); j++) {
				cout << "rects" << endl;
				cout << rects[j]["faceId"] << endl;
				cout << "person" << endl;
				cout << person[i]["faceId"] << endl;

				if (rects[j]["faceId"] == person[i]["faceId"]) {
					cout << "found matching face" << endl;
					//Draw rect on the rectangle coordinates
					cout <<  rects[i]["faceRectangle"]["left"] << endl;
					int topX = rects[i]["faceRectangle"]["left"].asInt();
					int topY = rects[i]["faceRectangle"]["top"].asInt();
					Point top(topX, topY);

					int bottomX = rects[i]["faceRectangle"]["width"].asInt() + topX;
					int bottomY = rects[i]["faceRectangle"]["height"].asInt() + topY;
					Point bottom(bottomX, bottomY);

					rectangle(image, top, bottom, (255, 0, 0), 3);
					string outputFile = "rect_drawn" + rects[j]["faceId"].asString() + ".jpg";
					imwrite(outputFile, image);
				}
			}

		}
	}


	
	return 0;
}