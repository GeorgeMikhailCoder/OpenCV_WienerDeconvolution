#include "OpenCV_WienerDeconvolution.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;

//  void simpleDFT(InputArray A, OutputArray C)
//  {
//      C.create(abs(A.rows()) + 1, abs(A.cols()) + 1, A.type());
//      Size dftSize;
//      // вычисляем размер преобразования ДПФ
//      dftSize.width = getOptimalDFTSize(A.cols() - 1);
//      dftSize.height = getOptimalDFTSize(A.rows()- 1);
//      Mat tmpA(dftSize, A.type(), Scalar::all(0));
//      Mat roiA(tmpA, Rect(0, 0, A.cols(), A.rows()));
//      A.copyTo(roiA);
//      dft(tmpA, tmpA, 0, A.rows());
//      tmpA(Rect(0, 0, C.cols(), C.rows())).copyTo(C);
//  }
//  
//  void convolveDFT(InputArray A, InputArray B, OutputArray C)
//  {
//      // при необходимости перераспределяем выходной массив
//      C.create(abs(A.rows() - B.rows()) + 1, abs(A.cols() - B.cols()) + 1, A.type());
//      Size dftSize;
//      // вычисляем размер преобразования ДПФ
//      dftSize.width = getOptimalDFTSize(A.cols() + B.cols() - 1);
//      dftSize.height = getOptimalDFTSize(A.rows() + B.rows() - 1);
//      int atype = A.type();
//      // выделяем временные буферы и инициализируем их нулями
//      Mat tmpA(dftSize, A.type(), Scalar::all(0));
//      Mat tmpB(dftSize, B.type(), Scalar::all(0));
//      // копируем A и B в левый верхний угол tempA и tempB соответственно
//      Mat roiA(tmpA, Rect(0, 0, A.cols(), A.rows()));
//      A.copyTo(roiA);
//      Mat roiB(tmpB, Rect(0, 0, B.cols(), B.rows()));
//      B.copyTo(roiB);
//      // теперь преобразуем дополненные A и B на месте;
//      // использовать подсказку "nonzeroRows" для более быстрой обработки
//      dft(tmpA, tmpA, 0, A.rows());
//      dft(tmpB, tmpB, 0, B.rows());
//      // перемножить спектры;
//      // функция хорошо обрабатывает упакованные представления спектра
//      
//      mulSpectrums(tmpA, tmpB, tmpA,0);
//      // преобразование продукта обратно из частотной области.
//      // Даже если все строки результатов будут ненулевыми,
//      // вам нужны только первые C.rows из них, поэтому вы
//      // передать nonzeroRows == C.rows
//      dft(tmpA, tmpA, DFT_INVERSE + DFT_SCALE, C.rows());
//      // теперь скопируйте результат обратно в C.
//      tmpA(Rect(0, 0, C.cols(), C.rows())).copyTo(C);
//      // все временные буферы будут автоматически освобождены
//  }

Mat DFT(const char* filename)
{
    Mat I = imread(filename, IMREAD_GRAYSCALE);
    if (I.empty())
    {
        Mat emty(7, 7, CV_32FC2, Scalar(1, 3));
        return emty;
    }

    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize(I.rows);
    int n = getOptimalDFTSize(I.cols); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);



    normalize(magI, magI, 0, 1); // Transform the matrix with float values into a
    // viewable image form (float between values 0 and 1).

    imshow("Input Image", I);    // Show the result
    imshow(filename, magI);
    //   waitKey();

    return magI;
}

Mat IDFT(Mat src)
{
    Mat I = src;
    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize(I.rows);
    int n = getOptimalDFTSize(I.cols); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI, DFT_INVERSE);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(IDFT(I))^2 + Im(IDFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(IDFT(I), planes[1] = Im(IDFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);


    normalize(magI, magI, 0, 1);

    imshow("forged map", magI);


    return magI;
}


#include <opencv2/core/core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
    Mat I(200, 300, CV_8UC3, Scalar(0));
    if (I.empty())
        return -1;

    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize(I.rows);
    int n = getOptimalDFTSize(I.cols); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);


    normalize(magI, magI, 0, 1); // Transform the matrix with float values into a
    // viewable image form (float between values 0 and 1).

    imshow("Input Image", I);    // Show the result
    imshow("Spectrum Magnitude", magI);
    waitKey();

    //calculating the idft
    cv::Mat inverseTransform;
    cv::dft(complexI, inverseTransform, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);
    normalize(inverseTransform, inverseTransform, 0, 1);
    imshow("Reconstructed", inverseTransform);
    waitKey();
    system("pause");
    return 0;
}