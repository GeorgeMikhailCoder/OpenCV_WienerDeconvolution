#include "OpenCV_WienerDeconvolution.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;


Mat toFloat(const Mat& src)
{
    Mat res;
    src.convertTo(res, CV_32FC3);
    res /= 255;
    return res;
}

Mat myDFT(Mat src)
{
    Mat b;
    dft(src, b, DFT_COMPLEX_OUTPUT);
    vector<Mat> mass;
    split(b, mass);
    Mat rf;
    rf = mass[0];
    return rf;
}

int main()
{
    Mat I(200, 300, CV_8UC3, Scalar(0));
    if (I.empty())
        return -1;

    Mat a(200, 300, CV_32F, Scalar(0.7));
    //  for (int y = 0; y < a.rows; y++)
    //  {
    //      for (int x = 0; x < a.cols; x++)
    //      {
    //          if (x > 100)
    //              a.at<float>(y, x) = 1.0;
    //          else
    //              a.at<float>(y, x) = 0.0;
    //          
    //          
    //          //  float sum = 0;
    //          //  for (float omega = 0.; omega < 5; omega += 1)
    //          //  {
    //          //      sum += sin(2 * 3.14 * omega * y / 200) + sin(2 * 3.14 * omega * x / 300);
    //          //  }
    //          //  a.at<float>(y, x) = sum;
    //      }
    //  }

    a = imread("../../../img/redcar1.jpg", IMREAD_GRAYSCALE);
    if (a.empty())
        return -1;

    a = toFloat(a);
    Mat b;

    imshow("a", a);
    Mat rf = myDFT(a);
    dft(a, b, DFT_COMPLEX_OUTPUT);

    imshow("f", rf);
    idft(b, b, DFT_REAL_OUTPUT | DFT_SCALE);

    imshow("b", b);
    waitKey();
    system("pause");
    cout << "Hello CMake." << endl;
    waitKey();
    system("pause");
    return 0;
}