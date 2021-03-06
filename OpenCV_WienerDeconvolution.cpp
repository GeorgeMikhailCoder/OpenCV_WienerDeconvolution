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

Mat conj(Mat src)
{
    vector<Mat> mass;
    split(src, mass);

    mass[1] = -mass[1];
    Mat dst;
    merge(mass, dst);
    return dst;
}


void checkSizeType(const Mat& A, const Mat& B)
{
    assert(A.rows == B.rows && A.cols == B.cols && A.depth() == B.depth() && A.channels() == B.channels());
}

Mat complMul(const Mat& A, const Mat& B)
{
    assert(A.channels() == 2);
    checkSizeType(A, B);
    
    Mat res(A.rows, A.cols, CV_32FC2);

    for (int y = 0; y < A.rows; y++)
    {
        for (int x = 0; x < A.cols; x++)
        {
            Vec2f a = A.at<Vec2f>(y, x);
            Vec2f b = B.at<Vec2f>(y, x);

            Vec2f c;
            c[0] = a[0] * b[0] - a[1] * b[1];
            c[1] = a[0] * b[1] + a[1] * b[0];

            res.at<Vec2f>(y, x) = c;
        }
    }

    return res;
}

Mat absSqr(const Mat& src)
{
    assert(src.channels() == 2);

    Mat res(src.rows, src.cols, CV_32FC2);

    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            Vec2f a = src.at<Vec2f>(y, x);

            Vec2f c;
            c[0] = a[0] * a[0] + a[1]*a[1];

            res.at<Vec2f>(y, x) = c;
        }
    }

    return res;
}

int main()
{
    Mat a;
    a = imread("../../../img/redcar1.jpg", IMREAD_GRAYSCALE);
    if (a.empty())
        return -1;

    a = toFloat(a);
    imshow("a", a);

    Mat u, U, h, H, Y, y,A,B,b;

    h = getGaussianKernel(5, 1);
    h = h * h.t();
    filter2D(a, u, -1, h, Point(-1, -1), 0, BORDER_DEFAULT);
    imshow("u",u);

    //int cy(h.rows / 2 + h.rows % 2), cx(h.cols / 2 + h.cols % 2);
    int cy(h.rows / 2), cx(h.cols / 2);

    Mat h_expand(u.rows,u.cols,u.depth(), Scalar(0));
    for(int y=0;y<h.rows;y++)
        for (int x = 0; x < h.cols; x++)
        {
            int newY = (h_expand.rows + cy - y) % h_expand.rows;
            int newX = (h_expand.cols + cx - x) % h_expand.cols;
            float tmp =(float) h.at<double>(y, x);
            h_expand.at<float>(newY, newX) = tmp;
        }

    dft(a, A, DFT_COMPLEX_OUTPUT);
    dft(u, U, DFT_COMPLEX_OUTPUT);
    //  cout << h<< endl;
    //  cout << h_expand << endl;
    dft(h_expand, H, DFT_COMPLEX_OUTPUT);
    // cout << H.rows << H.cols << endl;
    

    B = complMul(A, H);
    idft(B, b, DFT_REAL_OUTPUT | DFT_SCALE);
    imshow("b", b);
 
    Y.create(U.rows, U.cols, CV_32FC2);

    //  float  N = 0.001;
    //  Y = complMul(U, conj(H)) / (absSqr(H) + N);

    for (int y = 0; y < U.rows; y++)
        for (int x = 0; x < U.cols; x++)
        {
            float noise = 0.0000001;
    
            Vec2f uk = U.at<Vec2f>(y, x);
            Vec2f hk = H.at<Vec2f>(y, x);


            Vec2f hkc = hk;
            hkc[1] = -hkc[1];
    
            Vec2f chisl;
            chisl[0] = uk[0] * hkc[0] - uk[1] * hkc[1];
            chisl[1] = uk[0]*hkc[1] + uk[1] * hkc[0];
            
            float modh2 = hk[0] * hk[0] + hk[1] * hk[1];
            float znam = modh2 + noise;
    
            Vec2f res;
            res[0] = chisl[0] / znam;
            res[1] = chisl[1] / znam;
    
            Y.at<Vec2f>(y, x) = res;
        }
    

    idft(Y, y, DFT_REAL_OUTPUT | DFT_SCALE);

    imshow("y", y);
    waitKey();
    system("pause");
    return 0;
}