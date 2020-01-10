//
//  ViewController.m
//  BarCodesScanner
//
//  Created by qiudong on 2019/11/25.
//  Copyright © 2019 qiudong. All rights reserved.
//

#import "ViewController.h"
#import "ScanResultsTableViewController.h"
#import "kMeansCluster.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "zbar.h"
#import <CoreGraphics/CoreGraphics.h>

//#define GRAPH_PROCESS_LAB

#define AUTO_SELECT_SCANNING_THETA
#define SCAN_ORIGINAL_GRAY_IMAGE

static ViewController* g_viewController;

@interface ViewController ()
<
UITableViewDataSource,
UITableViewDelegate,
UINavigationControllerDelegate,
UIImagePickerControllerDelegate
>
{
    cv::Mat _sourceCVImage;
    cv::Mat _grayCVImage;
    cv::Mat _enhancedGrayCVImage;
    
    float _lineNormalTheta;
//    float _lineRHO;
}

@property (nonatomic, strong) IBOutlet UIImageView* imageView0;
@property (nonatomic, strong) IBOutlet UIImageView* imageView1;

@property (nonatomic, strong) IBOutlet UISlider* ratioSlider;

@property (nonatomic, strong) IBOutlet UILabel* logLabel;

@property (nonatomic, strong) IBOutlet UISlider* binaryThresholdSlider;
@property (nonatomic, strong) IBOutlet UISlider* medianBlurSizeSlider;
@property (nonatomic, strong) IBOutlet UISlider* thetaSlider;

@property (nonatomic, strong) IBOutlet UITableView* tableView;

@property (nonatomic, strong) NSMutableArray<NSString* >* imageTitles;
@property (nonatomic, strong) NSMutableArray<UIImage* >* images;

@property (nonatomic, strong) NSMutableArray<UIImage* >* barcodeImages;
@property (nonatomic, strong) NSMutableArray<NSString* >* barcodeNames;

-(IBAction) onSelectImageButtonClicked:(id)sender;

-(IBAction) onRatioSliderValueChanged:(id)sender;

-(IBAction) onScanButtonClicked:(id)sender;

-(IBAction) onBinaryThresholdSliderValueChanged:(id)sender;
-(IBAction) onMedianBlurSizeSliderValueChanged:(id)sender;
-(IBAction) onThetaSliderValueChanged:(id)sender;

-(void) appendImage:(UIImage*)image title:(NSString*)title;

@end

using namespace std;
using namespace zbar;
using namespace cv;
 
cv::Mat cvMatFromUIImage(UIImage* image)
{
  CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
  CGFloat cols = image.size.width;
  CGFloat rows = image.size.height;

  cv::Mat cvMat(rows, cols, CV_8UC4);// 8 bits per component, 4 channels (color channels + alpha)

  CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                 cols,                       // Width of bitmap
                                                 rows,                       // Height of bitmap
                                                 8,                          // Bits per component
                                                 cvMat.step[0],              // Bytes per row
                                                 colorSpace,                 // Colorspace
                                                 kCGImageAlphaNoneSkipLast |
                                                 kCGBitmapByteOrderDefault); // Bitmap info flags

  CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
  CGContextRelease(contextRef);

  return cvMat;
}

cv::Mat cvMatGrayFromUIImage(UIImage* image)
{
  CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
  CGFloat cols = image.size.width;
  CGFloat rows = image.size.height;

  cv::Mat cvMat(rows, cols, CV_8UC1); // 8 bits per component, 1 channels

  CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to data
                                                 cols,                       // Width of bitmap
                                                 rows,                       // Height of bitmap
                                                 8,                          // Bits per component
                                                 cvMat.step[0],              // Bytes per row
                                                 colorSpace,                 // Colorspace
                                                 kCGImageAlphaNoneSkipLast |
                                                 kCGBitmapByteOrderDefault); // Bitmap info flags

  CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
  CGContextRelease(contextRef);

  return cvMat;
}
//After the processing we need to convert it back to UIImage. The code below can handle both gray-scale and color image conversions (determined by the number of channels in the if statement).
//
//cv::Mat greyMat;
//cv::cvtColor(inputMat, greyMat, CV_BGR2GRAY);
UIImage* UIImageFromCVMat(cv::Mat cvMat)
{
  NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
  CGColorSpaceRef colorSpace;

  if (cvMat.elemSize() == 1) {
      colorSpace = CGColorSpaceCreateDeviceGray();
  } else {
      colorSpace = CGColorSpaceCreateDeviceRGB();
  }

  CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);

  // Creating CGImage from cv::Mat
  CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
                                     cvMat.rows,                                 //height
                                     8,                                          //bits per component
                                     8 * cvMat.elemSize(),                       //bits per pixel
                                     cvMat.step[0],                            //bytesPerRow
                                     colorSpace,                                 //colorspace
                                     kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
                                     provider,                                   //CGDataProviderRef
                                     NULL,                                       //decode
                                     false,                                      //should interpolate
                                     kCGRenderingIntentDefault                   //intent
                                     );


  // Getting UIImage from CGImage
  UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
  CGImageRelease(imageRef);
  CGDataProviderRelease(provider);
  CGColorSpaceRelease(colorSpace);

  return finalImage;
}

void showImage(Mat& mat, NSString* title) {
//    imshow(title.UTF8String, image);
    
    UIImage* img = UIImageFromCVMat(mat);
//    NSString* outPath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0] stringByAppendingPathComponent:[NSString stringWithFormat:@"%@.png", title]];
//    NSData* pngData = UIImagePNGRepresentation(img);
//    [pngData writeToFile:outPath
//              atomically:NO];
    
    [g_viewController appendImage:img title:title];
}

void plot(Mat& img, int x, int y, const Scalar& color) {
    if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) return;
//    int type = img.type();
//    int depth = img.depth();
//    int channels = img.channels();
//    NSLog(@"srcImg.(type, depth, channels) = (%d, %d, %d)", type, depth, channels);
    uchar* dstPtr = img.data + img.step[0] * y + img.step[1] * x;
    if (1 == img.channels())
    {
        dstPtr[0] = (int)((color[0] + color[1] + color[2])) & 0xff;
    }
    else if (3 == img.channels())
    {
        dstPtr[0] = (int)color[0] & 0xff;
        dstPtr[1] = (int)color[1] & 0xff;
        dstPtr[2] = (int)color[2] & 0xff;
    }
}

typedef void(*TrackCallback2D)(int x, int y, void* context);

void BresenhamLineTrack(cv::Point pt0, cv::Point pt1, TrackCallback2D callback, void* context) {
    int deltay = abs(pt1.y - pt0.y);
    int deltax = abs(pt1.x - pt0.x);
    int y = pt0.y, x = pt0.x;
    int ystep = pt0.y < pt1.y ? 1 : -1;
    int xstep = pt0.x < pt1.x ? 1 : -1;
    if (deltay > deltax)
    {
        int error = deltay / 2;
        for (int i = deltay; i >= 0; --i)
        {
            if (callback)
            {
                callback(x, y, context);
            }
            error -= deltax;
            if (error < 0)
            {
                x += xstep;
                error += deltay;
            }
            y += ystep;
        }
    }
    else
    {
        int error = deltax / 2;
        for (int i = deltax; i >= 0; --i)
        {
            if (callback)
            {
                callback(x, y, context);
            }
            error -= deltay;
            if (error < 0)
            {
                y += ystep;
                error += deltax;
            }
            x += xstep;
        }
    }
}

void BresenhamLineDrawCallback(int x, int y, void* context) {
    void** userData = (void**)context;
    Mat* pImg = (Mat*) userData[0];
    Scalar* pColor = (Scalar*) userData[1];
    plot(*pImg, x, y, *pColor);
}

void BresenhamLineDraw(Mat& img, cv::Point pt0, cv::Point pt1, const Scalar& color) {
    void** userData = new void*[2];
    userData[0] = &img;
    userData[1] = const_cast<Scalar*>(&color);
    BresenhamLineTrack(pt0, pt1, BresenhamLineDrawCallback, userData);
    delete[] userData;
}

void BresenhamLinePickCallback(int x, int y, void* context) {
    void** userData = (void**)context;
    Mat* pSrcImg = (Mat*) userData[0];
    if (x < 0 || x >= pSrcImg->cols || y < 0 || y >= pSrcImg->rows) return;
    uchar* srcPtr = pSrcImg->data + pSrcImg->step[0] * y + pSrcImg->step[1] * x;
    uchar* dstPtr = (uchar*) userData[1];
    memcpy(dstPtr, srcPtr, pSrcImg->channels());
    dstPtr += pSrcImg->channels();
    userData[1] = dstPtr;
}

void BresenhamLinePick(uchar* dstPtr, Mat& srcImg, cv::Point pt0, cv::Point pt1) {
    void** userData = new void*[2];
    userData[0] = &srcImg;
    userData[1] = dstPtr;
    BresenhamLineTrack(pt0, pt1, BresenhamLinePickCallback, userData);
    delete[] userData;
}

Mat rasterImageWithDirection(Mat& srcImg, float lineNormalTheta, float lineRHO) {
    int w = srcImg.cols, h = srcImg.rows;
    int diagonalLength = (int) ceil(sqrt(w * w + h * h));
    Mat dstImg(w + h + 2, diagonalLength, srcImg.type());
    float t = tan(lineNormalTheta), cosTheta = cos(lineNormalTheta), sinTheta = sin(lineNormalTheta);
    int px0, py0, px1, py1, dpx, dpy;
    if (0 <= t && t < 1)
    {
        px0 = 0;
        py0 = 0;
        px1 = w + ceil(h * t) - 1;
        py1 = 0;
        dpx = 1;
        dpy = 0;
    }
    else if (1 <= t)
    {
        px0 = 0;
        py0 = 0;
        px1 = 0;
        py1 = h + ceil(w / t) - 1;
        dpx = 0;
        dpy = 1;
    }
    else if (-1 <= t && t < 0)
    {
        px0 = floor(h * t);
        py0 = 0;
        px1 = w - 1;
        py1 = 0;
        dpx = 1;
        dpy = 0;
    }
    else// if (t < -1)
    {
        px0 = 0;
        py0 = floor(w / t);
        px1 = 0;
        py1 = h - 1;
        dpx = 0;
        dpy = 1;
    }
//    void** userData = new void*[2];
//    userData[0] = &srcImg;
//    userData[1] = dstPtr;
    int dstRow = 0;
    int px, py;
    for (px = px0, py = py0;
         px <= px1 && py <= py1;
         px += dpx, py += dpy, dstRow++)
    {
        cv::Point pt0, pt1;
        pt0.x = cvRound(px + diagonalLength * (-sinTheta));
        pt0.y = cvRound(py + diagonalLength * (cosTheta));
        pt1.x = cvRound(px - diagonalLength * (-sinTheta));
        pt1.y = cvRound(py - diagonalLength * (cosTheta));
        uchar* dstPtr = dstImg.data + dstImg.step[0] * dstRow;
        BresenhamLinePick(dstPtr, srcImg, pt0, pt1);
    }
    return dstImg;
}

Mat rasterImageWithDirectionOneLine(Mat& srcImg, float lineNormalTheta, float ratio) {
    int w = srcImg.cols, h = srcImg.rows;
    int diagonalLength = (int) ceil(sqrt(w * w + h * h));
    Mat oneLineImg(1, diagonalLength, srcImg.type());
    float t = tan(lineNormalTheta), cosTheta = cos(lineNormalTheta), sinTheta = sin(lineNormalTheta);
    int px0, py0, px1, py1, dpx, dpy;
    if (0 <= t && t < 1)
    {
        px0 = 0;
        py0 = 0;
        px1 = w + ceil(h * t) - 1;
        py1 = 0;
        dpx = 1;
        dpy = 0;
    }
    else if (1 <= t)
    {
        px0 = 0;
        py0 = 0;
        px1 = 0;
        py1 = h + ceil(w / t) - 1;
        dpx = 0;
        dpy = 1;
    }
    else if (-1 <= t && t < 0)
    {
        px0 = floor(h * t);
        py0 = 0;
        px1 = w - 1;
        py1 = 0;
        dpx = 1;
        dpy = 0;
    }
    else// if (t < -1)
    {
        px0 = 0;
        py0 = floor(w / t);
        px1 = 0;
        py1 = h - 1;
        dpx = 0;
        dpy = 1;
    }
    int px = px0 + (px1 - px0) * ratio;
    int py = py0 + (py1 - py0) * ratio;
    cv::Point pt0, pt1;
    pt0.x = cvRound(px + diagonalLength * (-sinTheta));
    pt0.y = cvRound(py + diagonalLength * (cosTheta));
    pt1.x = cvRound(px - diagonalLength * (-sinTheta));
    pt1.y = cvRound(py - diagonalLength * (cosTheta));
    uchar* dstPtr = oneLineImg.data;
    if (pt0.x <= pt1.x)
        BresenhamLinePick(dstPtr, srcImg, pt0, pt1);
    else
        BresenhamLinePick(dstPtr, srcImg, pt1, pt0);
    Mat expandedImg(31, diagonalLength, srcImg.type());
    copyMakeBorder(oneLineImg, expandedImg, 15, 15, 0, 0, BORDER_REPLICATE);
    return expandedImg;
}

Mat drawImageWithHoughLine(Mat& srcImg, float lineNormalTheta, float ratio) {
    int w = srcImg.cols, h = srcImg.rows;
    int diagonalLength = (int) ceil(sqrt(w * w + h * h));
    Mat dstImg;
    srcImg.copyTo(dstImg);
    float t = tan(lineNormalTheta), cosTheta = cos(lineNormalTheta), sinTheta = sin(lineNormalTheta);
    int px0, py0, px1, py1, dpx, dpy;
    if (0 <= t && t < 1)
    {
        px0 = 0;
        py0 = 0;
        px1 = w + ceil(h * t) - 1;
        py1 = 0;
        dpx = 1;
        dpy = 0;
    }
    else if (1 <= t)
    {
        px0 = 0;
        py0 = 0;
        px1 = 0;
        py1 = h + ceil(w / t) - 1;
        dpx = 0;
        dpy = 1;
    }
    else if (-1 <= t && t < 0)
    {
        px0 = floor(h * t);
        py0 = 0;
        px1 = w - 1;
        py1 = 0;
        dpx = 1;
        dpy = 0;
    }
    else// if (t < -1)
    {
        px0 = 0;
        py0 = floor(w / t);
        px1 = 0;
        py1 = h - 1;
        dpx = 0;
        dpy = 1;
    }
    int px = px0 + (px1 - px0) * ratio;
    int py = py0 + (py1 - py0) * ratio;
    cv::Point pt0, pt1;
    pt0.x = cvRound(px + diagonalLength * (-sinTheta));
    pt0.y = cvRound(py + diagonalLength * (cosTheta));
    pt1.x = cvRound(px - diagonalLength * (-sinTheta));
    pt1.y = cvRound(py - diagonalLength * (cosTheta));
    BresenhamLineDraw(dstImg, pt0, pt1, Scalar(255, 0, 0));
    return dstImg;
}

void rasterImageWithDirectionLineByLine(Mat& srcImg, float lineNormalTheta, void(^callback)(Mat&, BOOL)) {
    int w = srcImg.cols, h = srcImg.rows;
    int diagonalLength = (int) ceil(sqrt(w * w + h * h));
    Mat oneLineImg(1, diagonalLength, srcImg.type());
//    Mat expandedImg(121, diagonalLength, srcImg.type());
    float t = tan(lineNormalTheta), cosTheta = cos(lineNormalTheta), sinTheta = sin(lineNormalTheta);
    int px0, py0, px1, py1, dpx, dpy;
    if (0 <= t && t < 1)
    {
        px0 = 0;
        py0 = 0;
        px1 = w + ceil(h * t) - 1;
        py1 = 0;
        dpx = 1;
        dpy = 0;
    }
    else if (1 <= t)
    {
        px0 = 0;
        py0 = 0;
        px1 = 0;
        py1 = h + ceil(w / t) - 1;
        dpx = 0;
        dpy = 1;
    }
    else if (-1 <= t && t < 0)
    {
        px0 = floor(h * t);
        py0 = 0;
        px1 = w - 1;
        py1 = 0;
        dpx = 1;
        dpy = 0;
    }
    else// if (t < -1)
    {
        px0 = 0;
        py0 = floor(w / t);
        px1 = 0;
        py1 = h - 1;
        dpx = 0;
        dpy = 1;
    }
//    void** userData = new void*[2];
//    userData[0] = &srcImg;
//    userData[1] = dstPtr;
    int dstRow = 0;
    int px, py;
    for (px = px0, py = py0;
         px <= px1 && py <= py1;
         dstRow++)
    {
        cv::Point pt0, pt1;
        pt0.x = cvRound(px + diagonalLength * (-sinTheta));
        pt0.y = cvRound(py + diagonalLength * (cosTheta));
        pt1.x = cvRound(px - diagonalLength * (-sinTheta));
        pt1.y = cvRound(py - diagonalLength * (cosTheta));
        uchar* dstPtr = oneLineImg.data;
        BresenhamLinePick(dstPtr, srcImg, pt0, pt1);
        Mat expandedImg(31, diagonalLength, srcImg.type());
        copyMakeBorder(oneLineImg, expandedImg, 15, 15, 0, 0, BORDER_REPLICATE);
        
        px += dpx;
        py += dpy;
        if (callback)
        {
            callback(expandedImg, px > px1 || py > py1);
        }
    }
}

Mat HoughLineImage(Mat& srcGrayImage, Mat& enhancedGrayImage, vector<Vec2f>** pOutLines = NULL, int** pOutClusters = NULL) {
    Mat imageGaussian, imageSobelX, imageSobelY, imageSobelOut;
    GaussianBlur(srcGrayImage, imageGaussian, cv::Size(3,3), 0);
    //水平和垂直方向灰度图像的梯度和,使用Sobel算子
    Mat imageX16S, imageY16S;
    Sobel(imageGaussian, imageX16S, CV_16S, 1,0,3,1,0,4);
    Sobel(imageGaussian, imageY16S, CV_16S, 0,1,3,1,0,4);
    convertScaleAbs(imageX16S, imageSobelX, 1, 0);
    convertScaleAbs(imageY16S, imageSobelY, 1, 0);
    imageSobelOut = imageSobelX+imageSobelY;
    
    Mat erodedImg(imageSobelOut.size(), CV_8UC1);
    Mat dilateImg(imageSobelOut.size(), CV_8UC1);
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(23, 23), cv::Point(-1,-1));
    dilate(imageSobelOut, dilateImg, morphKernel, cv::Point(-1,-1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());
    erode(dilateImg, erodedImg, morphKernel, cv::Point(-1,-1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());
    
    Mat diffImg = dilateImg - erodedImg;
    threshold(diffImg, diffImg, 96, 255, THRESH_BINARY);
    
    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;
    findContours(diffImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    Mat contoursImg = Mat::zeros(diffImg.rows
                                 , diffImg.cols, CV_8UC1);
    long maxContourPoints[4] = {0};
    if (hierarchy.size() > 0)
    {
        for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
        {
            for (int i = sizeof(maxContourPoints) / sizeof(maxContourPoints[0]) - 1; i >= 0; --i)
            {
                if (contours.size() > 0 && contours[idx].size() >= maxContourPoints[i])
                {
                    for (int j=0; j<i; j++)
                    {
                        maxContourPoints[j] = maxContourPoints[j+1];
                    }
                    maxContourPoints[i] = contours[idx].size();
                    break;
                }
            }
        }
        NSLog(@"Top %ld longest contours' point counts : %ld, %ld", sizeof(maxContourPoints)/sizeof(maxContourPoints[0]), maxContourPoints[1], maxContourPoints[0]);
        for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
        {
            if (contours.size() <= 0 || contours[idx].size() < maxContourPoints[0]) continue;
    //        Scalar color(rand() & 255, rand() & 255, rand() & 255);
            float grayValueF = (float)contours[idx].size() / maxContourPoints[1];
            int grayValueI = (int)grayValueF * 255;
            Scalar color(grayValueI, grayValueI, grayValueI);
            drawContours(contoursImg, contours, idx, color, 1, LINE_8, hierarchy, -1);
        }
    }
    
    //HoughLines查找直线，该直线跟原图的一维码方向相互垂直
    vector<Vec2f>* outLines = new vector<Vec2f>;
    if (pOutLines) *pOutLines = outLines;
    vector<Vec2f>& lines = *outLines;
    
    float pi180 = (float)CV_PI / 180;
//    Mat linImg(contoursImg.size(), CV_8UC3);
    Mat linImg;
    cvtColor(enhancedGrayImage, linImg, COLOR_GRAY2BGR);
    int maxLength = (int) ceilf(sqrt(linImg.rows * linImg.rows + linImg.cols + linImg.cols));
    Mat linePickerImg(maxLength, maxLength, linImg.type());
    HoughLines(contoursImg, lines, 1, pi180, 128, 0, 0);
    long numLines = lines.size();
    NSLog(@"numLines = %ld", numLines);
//    numLines = 54;
    
    float** thetas = (float**) malloc(sizeof(float*) * numLines);
    for(int l=0; l<numLines; l++)
    {
        thetas[l] = (float*) malloc(sizeof(float));
        *thetas[l] = lines[l][1];
    }
    int* clusters = kMeansCluster((const float**)thetas, 1, (int)numLines, 4);
    for(int l=0; l<numLines; l++)
    {
        free(thetas[l]);
    }
    free(thetas);
    
    float theta;
    //*
    for(int l=0; l<numLines; l++)
    {
        float rho = lines[l][0];
        theta = lines[l][1];
//        float aa = (theta / CV_PI) * 180;
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        
        int cluster = clusters[l] + 1;
        int R = (cluster & 0x01) * 255;
        cluster >>= 1;
        int G = (cluster & 0x01) * 255;
        cluster >>= 1;
        int B = (cluster & 0x01) * 255;
        
        Scalar color = Scalar(R,G,B);
        line(linImg, pt1, pt2, color, 1, LINE_AA, 0);
        BresenhamLineDraw(linImg, pt1, pt2, color);
    }
    if (pOutClusters)
    {
        *pOutClusters = clusters;
    }
    else
    {
        free(clusters);///!!!
    }
    
    if (!pOutLines)
    {
        delete outLines;
    }
    return linImg;
}

Mat enhancedGrayImage(Mat& grayImg, float thresholdValue, float medianBlurSize) {
    Mat binaryImg, ret;
//    grayImg.copyTo(binaryImg);
//    threshold(grayImg, binaryImg, thresholdValue, 255, THRESH_BINARY);
//    medianBlur(binaryImg, ret, medianBlurSize);
    grayImg.copyTo(ret);
    return ret;
}

NSArray<NSDictionary* >* scanBarCodeInCVImg(Mat& scanImg) {
    ImageScanner scanner;
    scanner.set_config(ZBAR_CYCLIC, ZBAR_CFG_ENABLE, 1);
    scanner.set_config(ZBAR_CYCLIC, ZBAR_CFG_MIN_REPEATING_REQUIRED, 1);

    scanner.set_config(ZBAR_QRCODE, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_CODE39, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_CODE93, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_CODE128, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_CODABAR, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_DATABAR, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_DATABAR_EXP, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_I25, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_EAN2, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_EAN5, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_EAN8, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_EAN13, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_ISBN13, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_ISBN10, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_UPCA, ZBAR_CFG_ENABLE, 0);
    scanner.set_config(ZBAR_UPCE, ZBAR_CFG_ENABLE, 0);
    int width = scanImg.cols;
    int height = scanImg.rows;
    uchar* raw = (uchar*) scanImg.data;
    Image imageZbar(width, height, "Y800", raw, width * height);
    scanner.scan(imageZbar); //扫描条码
    Image::SymbolIterator symbol = imageZbar.symbol_begin();
    NSMutableArray<NSDictionary* >* results = [[NSMutableArray alloc] init];
    if (imageZbar.symbol_begin() == imageZbar.symbol_end())
    {
        results = nil;
    }
    for(; symbol != imageZbar.symbol_end(); ++symbol)
    {
        [results addObject:@{
            @"type": [NSString stringWithUTF8String:symbol->get_type_name().c_str()],
            @"barcode": [NSString stringWithUTF8String:symbol->get_data().c_str()],
        }];
    }
    imageZbar.set_data(NULL, 0);
    return results;
}

//void scanBarcodes(Mat& srcImg, float thresholdValue, float medianBlurSize, float theta, void(^scanCallback)(Mat& scanImg, NSString* resultInfo, BOOL isLastOne)) {
//    Mat imageGray;
////    cvtColor(srcImg, imageGray, COLOR_BGR2GRAY);
//    srcImg.copyTo(imageGray);
//    threshold(imageGray, imageGray, thresholdValue, 255, THRESH_BINARY);
//    medianBlur(imageGray, srcImg, medianBlurSize);
//    rasterImageWithDirectionLineByLine(srcImg, theta, ^(Mat& scanImg, BOOL isLastOne) {
//        NSArray<NSDictionary* >* results = scanBarCodeInCVImg(scanImg);
//
//        if (scanCallback)
//        {
//            scanCallback(scanImg, [results description], isLastOne);
//        }
//    });
//}

@interface ImageCell : UITableViewCell

@property (nonatomic, strong) IBOutlet UIImageView* imageView;
@property (nonatomic, strong) IBOutlet UILabel* titleLabel;

@end

@implementation ImageCell

@end


@implementation ViewController

-(void) appendImage:(UIImage*)image title:(NSString*)title {
    dispatch_async(dispatch_get_main_queue(), ^{
        [self.imageTitles addObject:title];
        [self.images addObject:image];
        [self.tableView reloadData];
    });
}

-(void) removeAllImages {
    dispatch_async(dispatch_get_main_queue(), ^{
        [self.imageTitles removeAllObjects];
        [self.images removeAllObjects];
        [self.tableView reloadData];
    });
}

//-(void) scanBarcodesInImage:(Mat&)srcImg thresholdValue:(float)thresholdValue medianBlurSize:(float)medianBlurSize theta:(float)theta {
//    NSMutableDictionary<NSString*, NSNumber* >* results = [[NSMutableDictionary alloc] init];
//    NSMutableDictionary<NSString*, UIImage* >* resultImages = [[NSMutableDictionary alloc] init];
//    scanBarcodes(srcImg, thresholdValue, medianBlurSize, theta, ^(Mat& scanImg, NSString* resultInfo, BOOL isLastOne) {
//        if (resultInfo)
//        {
//            NSNumber* count = results[resultInfo];
//            if (!count)
//            {
//                results[resultInfo] = @(1);
//            }
//            else
//            {
//                results[resultInfo] = @(count.intValue + 1);
//            }
//
//            UIImage* image = resultImages[resultInfo];
//            if (!image)
//            {
//                image = UIImageFromCVMat(scanImg);
//                resultImages[resultInfo] = image;
//            }
//        }
//        if (isLastOne)
//        {
//            dispatch_async(dispatch_get_main_queue(), ^{
//                self.logLabel.text = @"Done\n";
//            });
//            for (NSString* resultInfo in results.keyEnumerator)
//            {
//                UIImage* image = resultImages[resultInfo];
//                dispatch_async(dispatch_get_main_queue(), ^{
//                    self.imageView1.image = image;
//                    self.logLabel.text = [NSString stringWithFormat:@"%@\n%@",  self.logLabel.text, resultInfo];
//                });
//            }
//        }
//    });
//}

-(void) processImageV1:(NSString*)sourceImagePath {
    [self removeAllImages];
    Mat image, imageGray, imageGaussian;
    Mat imageSobelX, imageSobelY, imageSobelOut;
    image = imread(sourceImagePath.UTF8String, IMREAD_COLOR);//TODO
    cvtColor(image, imageGray, COLOR_BGR2GRAY);
    imageGray.copyTo(image);
//    threshold(imageGray, imageGray, 160, 255, THRESH_BINARY);
//    medianBlur(imageGray, image, 5);
//    threshold(imageGray, image, 164, 255, THRESH_BINARY);
    _sourceCVImage = image;
    showImage(image, [NSString stringWithFormat:@"Source - %@", sourceImagePath.lastPathComponent]);
    dispatch_async(dispatch_get_main_queue(), ^{
        g_viewController.imageView0.image = UIImageFromCVMat(image);
    });
    
    GaussianBlur(imageGray, imageGaussian, cv::Size(3,3), 0);
//    showImage(imageGray, [NSString stringWithFormat:@"Gray - %@", sourceImagePath.lastPathComponent]);
    //水平和垂直方向灰度图像的梯度和,使用Sobel算子
    Mat imageX16S, imageY16S;
    Sobel(imageGaussian, imageX16S, CV_16S, 1,0, 3, 1, 0, 4);
    Sobel(imageGaussian, imageY16S, CV_16S, 0,1, 3, 1, 0, 4);
    convertScaleAbs(imageX16S, imageSobelX, 1, 0);
    convertScaleAbs(imageY16S, imageSobelY, 1, 0);
    imageSobelOut = imageSobelX+imageSobelY;
    showImage(imageSobelOut, [NSString stringWithFormat:@"XY gradient - %@", sourceImagePath.lastPathComponent]);
    
    Mat erodedImg(imageSobelOut.size(), CV_8UC1);
    Mat dilateImg(imageSobelOut.size(), CV_8UC1);
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(23, 23), cv::Point(-1,-1));
    dilate(imageSobelOut, dilateImg, morphKernel, cv::Point(-1,-1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());
    showImage(dilateImg, [NSString stringWithFormat:@"Dilated - %@", sourceImagePath.lastPathComponent]);
    erode(dilateImg, erodedImg, morphKernel, cv::Point(-1,-1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());
    showImage(erodedImg, [NSString stringWithFormat:@"Then eroded - %@", sourceImagePath.lastPathComponent]);
    
    Mat diffImg = dilateImg - erodedImg;
    threshold(diffImg, diffImg, 96, 255, THRESH_BINARY);
    showImage(diffImg, [NSString stringWithFormat:@"Diff - %@", sourceImagePath.lastPathComponent]);
    
    medianBlur(diffImg, diffImg, 19);
    showImage(diffImg, [NSString stringWithFormat:@"Median blur - %@", sourceImagePath.lastPathComponent]);
    
    morphKernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(10, 10), cv::Point(-1,-1));
    erode(diffImg, diffImg, morphKernel, cv::Point(-1,-1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());
    showImage(diffImg, [NSString stringWithFormat:@"Erode again - %@", sourceImagePath.lastPathComponent]);
    
    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;
    findContours(diffImg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
    Mat contoursImg = Mat::zeros(diffImg.rows
                                 , diffImg.cols, CV_8UC1);
    long maxContourPoints[4] = {0};
    if (hierarchy.size() > 0)
    {
        for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
        {
            for (int i = sizeof(maxContourPoints) / sizeof(maxContourPoints[0]) - 1; i >= 0; --i)
            {
                if (contours.size() > 0 && contours[idx].size() >= maxContourPoints[i])
                {
                    for (int j=0; j<i; j++)
                    {
                        maxContourPoints[j] = maxContourPoints[j+1];
                    }
                    maxContourPoints[i] = contours[idx].size();
                    break;
                }
            }
        }
        NSLog(@"Top %ld longest contours' point counts : %ld, %ld", sizeof(maxContourPoints)/sizeof(maxContourPoints[0]), maxContourPoints[1], maxContourPoints[0]);
        for (int idx = 0; idx >= 0; idx = hierarchy[idx][0])
        {
            if (contours.size() <= 0 || contours[idx].size() < maxContourPoints[0]) continue;
    //        Scalar color(rand() & 255, rand() & 255, rand() & 255);
            float grayValueF = (float)contours[idx].size() / maxContourPoints[1];
            int grayValueI = (int)grayValueF * 255;
            Scalar color(grayValueI, grayValueI, grayValueI);
            drawContours(contoursImg, contours, idx, color, 1, LINE_8, hierarchy, -1);
        }
    }
    showImage(contoursImg, [NSString stringWithFormat:@"Contours - %@", sourceImagePath.lastPathComponent]);
    
    //HoughLines查找直线，该直线跟原图的一维码方向相互垂直
    vector<Vec2f> lines;
    float pi180 = (float)CV_PI / 180;
//    Mat linImg(contoursImg.size(), CV_8UC3);
    Mat linImg;
    image.copyTo(linImg);
    int maxLength = (int) ceilf(sqrt(linImg.rows * linImg.rows + linImg.cols + linImg.cols));
    Mat linePickerImg(maxLength, maxLength, linImg.type());
    HoughLines(contoursImg, lines, 1, pi180, 128, 0, 0);
//    HoughLines(diffImg, lines, 1, pi180, 128, 0, 0);
    long numLines = lines.size();
    NSLog(@"numLines = %ld", numLines);
//    numLines = 54;
    float theta;
    //*
    for(int l=0; l<numLines; l++)
    {
        float rho = lines[l][0];
        theta = lines[l][1];
//        float aa = (theta / CV_PI) * 180;
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line(linImg, pt1, pt2, Scalar(255,0,0), 1, LINE_AA, 0);
        BresenhamLineDraw(linImg, pt1, pt2, Scalar(0, 255, 0));
    }
    /*/
    cv::Point pt2 = cv::Point(linImg.cols / 2, 0), pt1 = cv::Point(linImg.cols, linImg.rows);
//    line(linImg, pt1, pt2, Scalar(255,0,0), 1, LINE_AA, 0);
    BresenhamLine(linImg, pt1, pt2, Scalar(255,0,0));
    //*/
    showImage(linImg, [NSString stringWithFormat:@"Hough lines - %@", sourceImagePath.lastPathComponent]);
    
    float angelD = 180 * theta / CV_PI - 90;
    cv::Point center(image.cols/2, image.rows/2);
    Mat rotMat = getRotationMatrix2D(center,angelD,1.0);
    Mat imageSource = Mat::ones(image.size(),CV_8UC3);
    warpAffine(image, imageSource, rotMat, image.size(), 1, 0, Scalar(255,255,255));//仿射变换校正图像
    showImage(linImg, [NSString stringWithFormat:@"Rotated - %@", sourceImagePath.lastPathComponent]);
//    _lineNormalTheta = lines[0][1];
//    _lineRHO = lines[0][0];
//    Mat rasterImg = rasterImageWithDirection(image, lines[0][1], lines[0][0]);
//    showImage(rasterImg, [NSString stringWithFormat:@"Raster with lines - %@", sourceImagePath.lastPathComponent]);
//    dispatch_async(dispatch_get_main_queue(), ^{
//        g_viewController.dstImageView.image = UIImageFromCVMat(rasterImg);
//    });
//    rasterImageWithDirectionLineByLine(image, lines[0][1], ^(Mat& img) {
//        Mat copiedImg = img;
//        dispatch_async(dispatch_get_main_queue(), ^{
//            NSLog(@"Set image");
//            g_viewController.dstImageView.image = UIImageFromCVMat(copiedImg);
//        });
//    });
    
    //校正角度计算
//    float angelD = 180 * theta / CV_PI - 90;
//    cv::Point center(image.cols/2, image.rows/2);
//    Mat rotMat = getRotationMatrix2D(center,angelD,1.0);
//    Mat imageSource = Mat::ones(image.size(),CV_8UC3);
//    warpAffine(image, imageSource, rotMat, image.size(), 1, 0, Scalar(255,255,255));//仿射变换校正图像
//    showImage(imageSource, [NSString stringWithFormat:@"Rotated - %@", sourceImagePath.lastPathComponent]);
    //Zbar一维码识别
//    ImageScanner scanner;
//    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
//    int width1 = imageSource.cols;
//    int height1 = imageSource.rows;
//    uchar *raw = (uchar *)imageSource.data;
//    Image imageZbar(width1, height1, "Y800", raw, width1 * height1);
//    scanner.scan(imageZbar); //扫描条码
//    Image::SymbolIterator symbol = imageZbar.symbol_begin();
//    if(imageZbar.symbol_begin()==imageZbar.symbol_end())
//    {
//        cout<<"查询条码失败，请检查图片！"<<endl;
//    }
//    for(;symbol != imageZbar.symbol_end();++symbol)
//    {
//        cout<<"类型："<<endl<<symbol->get_type_name()<<endl<<endl;
//        cout<<"条码："<<endl<<symbol->get_data()<<endl<<endl;
//    }
//    namedWindow("Source Window",0);
//    imshow("Source Window",imageSource);
//    waitKey();
//    imageZbar.set_data(NULL,0);
    
    /*
    Mat srcImg = image;
    int type = srcImg.type();
    int depth = srcImg.depth();
    int channels = srcImg.channels();
    NSLog(@"srcImg.(type, depth, channels) = (%d, %d, %d)", type, depth, channels);
    uchar* dstPtr = srcImg.data;
    long dRow = srcImg.rows / 8;
    int value = 0;
    long iRow;
    for (iRow = 0; iRow < srcImg.rows - dRow; iRow += dRow)
    {
        memset(dstPtr, value, srcImg.step * dRow);
        dstPtr += srcImg.step * dRow;
        value = (0 == value) ? 0xff : 0;
    }
    if (srcImg.rows - iRow < dRow / 2)
    {
        value = (0 == value) ? 0xff : 0;
    }
    memset(dstPtr, value, srcImg.step * (srcImg.rows - iRow));
    
    showImage(srcImg, [NSString stringWithFormat:@"Modified - %@", sourceImagePath.lastPathComponent]);
    //宽高扩充，非必须，特定的宽高可以提高傅里叶运算效率
    Mat padded;
    int opHeight = getOptimalDFTSize(srcImg.rows);
    int opWidth = getOptimalDFTSize(srcImg.cols);
    copyMakeBorder(srcImg, padded, 0, opHeight - srcImg.rows, 0, opWidth - srcImg.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat comImg;
    //通道融合，融合成一个2通道的图像
    merge(planes, 2, comImg);
    dft(comImg, comImg);
    split(comImg, planes);
      magnitude(planes[0], planes[1], planes[0]);
    Mat magMat = planes[0];
    magMat += Scalar::all(1);
    log(magMat, magMat);     //对数变换，方便显示
    magMat = magMat(cv::Rect(0, 0, magMat.cols & -2, magMat.rows & -2));
    //以下把傅里叶频谱图的四个角落移动到图像中心
    int cx = magMat.cols/2;
    int cy = magMat.rows/2;
    Mat q0(magMat, cv::Rect(0, 0, cx, cy));
    Mat q1(magMat, cv::Rect(0, cy, cx, cy));
    Mat q2(magMat, cv::Rect(cx, cy, cx, cy));
    Mat q3(magMat, cv::Rect(cx, 0, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q2.copyTo(q0);
    tmp.copyTo(q2);
    q1.copyTo(tmp);
    q3.copyTo(q1);
    tmp.copyTo(q3);
    normalize(magMat, magMat, 0, 1, NORM_MINMAX);
    Mat magImg(magMat.size(), CV_8UC1);
    magMat.convertTo(magImg, CV_8UC1, 255, 0);
    showImage(magImg, [NSString stringWithFormat:@"FFT - %@", sourceImagePath.lastPathComponent]);
    threshold(magImg, magImg, 180, 255, THRESH_BINARY);
    showImage(magImg, [NSString stringWithFormat:@"Binary - %@", sourceImagePath.lastPathComponent]);
    //*/
    
}

-(void) preprocessImage_backgroud:(NSString*)imagePath {///threshold:(float)threshold medianBlurSize:(float)medianBlurSize {
    _sourceCVImage = imread(imagePath.UTF8String, IMREAD_COLOR);
    cvtColor(_sourceCVImage, _grayCVImage, COLOR_BGR2GRAY);
    dispatch_async(dispatch_get_main_queue(), ^{
        [self updatePreprocessedImage];
    });
}

-(void) preprocessImage:(NSString*)imagePath {
//    int medianBlurSize = roundf(self.medianBlurSizeSlider.value);
//    if (medianBlurSize % 2 == 0)
//    {
//        medianBlurSize++;
//    }
//    float threshold = self.binaryThresholdSlider.value;
//    float theta = _thetaSlider.value * M_PI / 180.f;
//    float ratio = _ratioSlider.value;
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [self preprocessImage_backgroud:imagePath];/// threshold:threshold medianBlurSize:medianBlurSize];
//
//        [self updatePreprocessedImage_backgroud:threshold medianBlurSize:medianBlurSize theta:theta ratio:ratio];
    });
}

-(void) updatePreprocessedImage_backgroud:(float)threshold medianBlurSize:(float)medianBlurSize resultHandler:(void(^)(NSArray<NSDictionary* >* barcodeResults, UIImage* houghImage, UIImage* scannedImage, int progress, int total))resultHandler {
    _enhancedGrayCVImage = enhancedGrayImage(_grayCVImage, threshold, medianBlurSize);

    vector<Vec2f>* pLines = NULL;
    int* pClusters = NULL;
    Mat houghCVImage = HoughLineImage(_grayCVImage, _enhancedGrayCVImage, &pLines, &pClusters);

    float maxThetasOfCluster[4] = {0.f};
    float minThetasOfCluster[4] = {M_PI * 2};
    for (int i = (int)pLines->size() - 1; i >= 0; --i)
    {
        int cluster = pClusters[i];
        float theta = (*pLines)[i][1];
        if (theta > maxThetasOfCluster[cluster])
            maxThetasOfCluster[cluster] = theta;
        if (theta < minThetasOfCluster[cluster])
            minThetasOfCluster[cluster] = theta;
    }
    int mostFitCluster = 0;
    float minAbsCosTheta = 1.0f;
    float meanTheta = M_PI / 2.f;
    for (int i=0; i<4; ++i)
    {
        float absCosTheta = fabsf(cosf((maxThetasOfCluster[i] + minThetasOfCluster[i]) / 2.f));
        if (absCosTheta < minAbsCosTheta)
        {
            minAbsCosTheta = absCosTheta;
            mostFitCluster = i;
            meanTheta = (maxThetasOfCluster[i] + minThetasOfCluster[i]) / 2.f;
        }
    }
    float w = _sourceCVImage.cols, h = _sourceCVImage.rows;
    int diagonalLength = (int) ceil(sqrt(w * w + h * h));
    float t = tan(meanTheta);
    int d;
    if (0 <= t && t < 1)
    {
        d = w + ceil(h * t);
    }
    else if (1 <= t)
    {
        d = h + ceil(w / t);
    }
    else if (-1 <= t && t < 0)
    {
        d = w - floor(h * t);
    }
    else// if (t < -1)
    {
        d = h - floor(w / t);
    }
    ///!!!d /= 16;
    for (int progress = 0; progress <= d; progress += 1)
    {
        float ratio = (float)progress / (float)d;
        for (int i = (int)pLines->size() - 1; i >= 0; --i)
        {
            if (pClusters[i] != mostFitCluster) continue;
            float theta = (*pLines)[i][1];
#ifdef SCAN_ORIGINAL_GRAY_IMAGE
            Mat scannedCVImg = rasterImageWithDirectionOneLine(_grayCVImage, theta, ratio);
#else //#ifdef SCAN_ORIGINAL_GRAY_IMAGE
            Mat scannedCVImg = rasterImageWithDirectionOneLine(_enhancedGrayCVImage, theta, ratio);
        equalizeHist(scannedCVImg, scannedCVImg);
#endif //#ifdef SCAN_ORIGINAL_GRAY_IMAGE
            float tanTheta = tan(theta), cosTheta = cos(theta), sinTheta = sin(theta);
            int px0, py0, px1, py1, dpx, dpy;
            if (0 <= tanTheta && tanTheta < 1)
            {
                px0 = 0;
                py0 = 0;
                px1 = w + ceil(h * tanTheta) - 1;
                py1 = 0;
                dpx = 1;
                dpy = 0;
            }
            else if (1 <= tanTheta)
            {
                px0 = 0;
                py0 = 0;
                px1 = 0;
                py1 = h + ceil(w / tanTheta) - 1;
                dpx = 0;
                dpy = 1;
            }
            else if (-1 <= tanTheta && tanTheta < 0)
            {
                px0 = floor(h * tanTheta);
                py0 = 0;
                px1 = w - 1;
                py1 = 0;
                dpx = 1;
                dpy = 0;
            }
            else// if (t < -1)
            {
                px0 = 0;
                py0 = floor(w / tanTheta);
                px1 = 0;
                py1 = h - 1;
                dpx = 0;
                dpy = 1;
            }
            int px = px0 + (px1 - px0) * ratio;
            int py = py0 + (py1 - py0) * ratio;
            cv::Point pt0, pt1;
            pt0.x = cvRound(px + diagonalLength * (-sinTheta));
            pt0.y = cvRound(py + diagonalLength * (cosTheta));
            pt1.x = cvRound(px - diagonalLength * (-sinTheta));
            pt1.y = cvRound(py - diagonalLength * (cosTheta));
            
            NSArray<NSDictionary* >* results = scanBarCodeInCVImg(scannedCVImg);
//            dispatch_async(dispatch_get_main_queue(), ^{
                if (resultHandler)
                {
                    if (progress + 1 > d)
                    {
                        BresenhamLineDraw(houghCVImage, pt0, pt1, Scalar(255, 0, 0));
                        UIImage* houghImage = UIImageFromCVMat(houghCVImage);
                        UIImage* scannedImg = UIImageFromCVMat(scannedCVImg);
                        
                        resultHandler(results, houghImage, scannedImg, d, d);
                    }
                    else
                    {
                        BresenhamLineDraw(houghCVImage, pt0, pt1, Scalar(255, 0, 0));
                        UIImage* scannedImg = UIImageFromCVMat(scannedCVImg);
                        
                        resultHandler(results, nil, scannedImg, progress, d);
                    }
                }
//            });
            break;///!!!
        }

    }
    free(pClusters);
    delete pLines;
}

-(void) updatePreprocessedImage_backgroud:(float)threshold medianBlurSize:(float)medianBlurSize theta:(float)theta ratio:(float)ratio resultHandler:(void(^)(NSArray<NSDictionary* >* barcodeResults, UIImage* houghImage, UIImage* scannedImage))resultHandler {
#ifdef SCAN_ORIGINAL_GRAY_IMAGE
    Mat scannedCVImg = rasterImageWithDirectionOneLine(_grayCVImage, theta, ratio);
#else //#ifdef SCAN_ORIGINAL_GRAY_IMAGE
    Mat scannedCVImg = rasterImageWithDirectionOneLine(_enhancedGrayCVImage, theta, ratio);
    equalizeHist(scannedCVImg, scannedCVImg);
#endif //#ifdef SCAN_ORIGINAL_GRAY_IMAGE
    vector<Vec2f>* pLines = NULL;
    int* pClusters = NULL;
    Mat houghCVImage = HoughLineImage(_grayCVImage, _enhancedGrayCVImage, &pLines, &pClusters);
    {
        int w = houghCVImage.cols, h = houghCVImage.rows;
        int diagonalLength = (int) ceil(sqrt(w * w + h * h));
        float t = tan(theta), cosTheta = cos(theta), sinTheta = sin(theta);
        int px0, py0, px1, py1, dpx, dpy;
        if (0 <= t && t < 1)
        {
            px0 = 0;
            py0 = 0;
            px1 = w + ceil(h * t) - 1;
            py1 = 0;
            dpx = 1;
            dpy = 0;
        }
        else if (1 <= t)
        {
            px0 = 0;
            py0 = 0;
            px1 = 0;
            py1 = h + ceil(w / t) - 1;
            dpx = 0;
            dpy = 1;
        }
        else if (-1 <= t && t < 0)
        {
            px0 = floor(h * t);
            py0 = 0;
            px1 = w - 1;
            py1 = 0;
            dpx = 1;
            dpy = 0;
        }
        else// if (t < -1)
        {
            px0 = 0;
            py0 = floor(w / t);
            px1 = 0;
            py1 = h - 1;
            dpx = 0;
            dpy = 1;
        }
        int px = px0 + (px1 - px0) * ratio;
        int py = py0 + (py1 - py0) * ratio;
        cv::Point pt0, pt1;
        pt0.x = cvRound(px + diagonalLength * (-sinTheta));
        pt0.y = cvRound(py + diagonalLength * (cosTheta));
        pt1.x = cvRound(px - diagonalLength * (-sinTheta));
        pt1.y = cvRound(py - diagonalLength * (cosTheta));
        BresenhamLineDraw(houghCVImage, pt0, pt1, Scalar(255, 0, 0));
        
        UIImage* houghImage = UIImageFromCVMat(houghCVImage);
        UIImage* scannedImg = UIImageFromCVMat(scannedCVImg);
        NSArray<NSDictionary* >* results = scanBarCodeInCVImg(scannedCVImg);
        dispatch_async(dispatch_get_main_queue(), ^{
            if (resultHandler)
            {
                resultHandler(results, houghImage, scannedImg);
            }
        });
    }
    free(pClusters);
    delete pLines;
}

-(void) updatePreprocessedImage {
    if (_sourceCVImage.rows <= 0 || _sourceCVImage.cols <= 0) return;
    
    int medianBlurSize = roundf(self.medianBlurSizeSlider.value);
    if (medianBlurSize % 2 == 0)
    {
        medianBlurSize++;
    }
    float threshold = self.binaryThresholdSlider.value;
    float theta = _thetaSlider.value * M_PI / 180.f;
    float ratio = _ratioSlider.value;
   _enhancedGrayCVImage = enhancedGrayImage(_grayCVImage, threshold, medianBlurSize); dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [self updatePreprocessedImage_backgroud:threshold medianBlurSize:medianBlurSize theta:theta ratio:ratio resultHandler:^(NSArray<NSDictionary *> *barcodeResults, UIImage *houghImage, UIImage *scannedImage) {
            NSMutableString* resultStr = [[NSMutableString alloc] init];
            for (NSDictionary* result in barcodeResults)
            {
                [resultStr appendFormat:@"%@:%@, ", result[@"type"], result[@"barcode"]];
            }
            dispatch_async(dispatch_get_main_queue(), ^{
                self.imageView1.image = houghImage;
                self.imageView0.image = scannedImage;
                self.logLabel.text = resultStr;
            });
        }];
    });
}
#ifdef AUTO_SELECT_SCANNING_THETA
-(void) autoScanImage {
    if (_sourceCVImage.rows <= 0 || _sourceCVImage.cols <= 0) return;
    
    int medianBlurSize = roundf(self.medianBlurSizeSlider.value);
    if (medianBlurSize % 2 == 0)
    {
        medianBlurSize++;
    }
    float threshold = self.binaryThresholdSlider.value;
    
    _barcodeImages = [[NSMutableArray alloc] init];
    _barcodeNames = [[NSMutableArray alloc] init];
//    __block NSString* prevKey = nil;
    NSMutableSet* keys = [[NSMutableSet alloc] init];
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [self updatePreprocessedImage_backgroud:threshold medianBlurSize:medianBlurSize resultHandler:^(NSArray<NSDictionary *> *barcodeResults, UIImage *houghImage, UIImage *scannedImage, int progress, int total) {
            NSMutableString* resultStr = [[NSMutableString alloc] init];
            for (NSDictionary* result in barcodeResults)
            {
                [resultStr appendFormat:@"%@:%@, ", result[@"type"], result[@"barcode"]];
            }
            NSString* key = [NSString stringWithString:resultStr];
//            if (key && key.length > 0 && ![key isEqualToString:prevKey])
//            {
//                prevKey = key;
            if (key && key.length > 0 && ![keys containsObject:key])
            {
                [keys addObject:key];
                [self.barcodeNames addObject:key];
                [self.barcodeImages addObject:scannedImage];
            }
            
            if (progress >= total)
            {
                dispatch_async(dispatch_get_main_queue(), ^{
                    self.imageView1.image = houghImage;
                    self.imageView0.image = scannedImage;
                    self.logLabel.text = @"Done";
                    [self performSegueWithIdentifier:@"toResults" sender:nil];
                });
            }
            else
            {
                dispatch_async(dispatch_get_main_queue(), ^{
                    self.logLabel.text = [NSString stringWithFormat:@"Scanning... %d/%d", progress, total];
                });
            }
        }];
    });
}
#else //#ifdef AUTO_SELECT_SCANNING_THETA
-(void) autoScanImage {
    if (_sourceCVImage.rows <= 0 || _sourceCVImage.cols <= 0) return;
    
    int medianBlurSize = roundf(self.medianBlurSizeSlider.value);
    if (medianBlurSize % 2 == 0)
    {
        medianBlurSize++;
    }
    float threshold = self.binaryThresholdSlider.value;
    float theta = _thetaSlider.value * M_PI / 180.f;
    float t = tan(theta), w = _sourceCVImage.cols, h = _sourceCVImage.rows;
    int d;
    if (0 <= t && t < 1)
    {
        d = w + ceil(h * t);
    }
    else if (1 <= t)
    {
        d = h + ceil(w / t);
    }
    else if (-1 <= t && t < 0)
    {
        d = w - floor(h * t);
    }
    else// if (t < -1)
    {
        d = h - floor(w / t);
    }
    ///!!!d /= 16;
    
    _barcodeImages = [[NSMutableArray alloc] init];
    _barcodeNames = [[NSMutableArray alloc] init];
//    __block NSString* prevKey = nil;
    NSMutableSet* keys = [[NSMutableSet alloc] init];
    _enhancedGrayCVImage = enhancedGrayImage(_grayCVImage, threshold, medianBlurSize);
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        for (int progress = 0; progress <= d; progress += 1)
        {
            float ratio = (float)progress / (float)d;
            [self updatePreprocessedImage_backgroud:threshold medianBlurSize:medianBlurSize theta:theta ratio:ratio resultHandler:^(NSArray<NSDictionary *> *barcodeResults, UIImage *houghImage, UIImage *scannedImage) {
                NSMutableString* resultStr = [[NSMutableString alloc] init];
                for (NSDictionary* result in barcodeResults)
                {
                    [resultStr appendFormat:@"%@:%@, ", result[@"type"], result[@"barcode"]];
                }
                NSString* key = [NSString stringWithString:resultStr];
    //            if (key && key.length > 0 && ![key isEqualToString:prevKey])
    //            {
    //                prevKey = key;
                if (key && key.length > 0 && ![keys containsObject:key])
                {
                    [keys addObject:key];
                    [self.barcodeNames addObject:key];
                    [self.barcodeImages addObject:scannedImage];
                }
                
                if (progress == d)
                {
                    self.imageView1.image = houghImage;
                    self.imageView0.image = scannedImage;
                    self.logLabel.text = @"Done";
                    
                    [self performSegueWithIdentifier:@"toResults" sender:nil];
                }
                else
                {
                    self.logLabel.text = [NSString stringWithFormat:@"Scanning... %d/%d", progress, d];
                }
            }];
        }
    });
}
#endif //#ifdef AUTO_SELECT_SCANNING_THETA
-(void) viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    _images = [[NSMutableArray alloc] init];
    _imageTitles = [[NSMutableArray alloc] init];
//    NSArray* sourceFileNames = @[@"barcode.jpeg", @"barcodes.jpg", @"barcodes0.png", @"barcodes1.png", @"barcodes2.png", @"barcodes3.png"];
    NSArray* sourceFileNames = @[
#ifdef TEST_ZBAR
    @"barcode_sample_0.png",
#else
//    @"barcodes.png",
//    @"barcodes0.png",
//    @"barcodes1.png",
//    @"barcodes2.png",
//    @"infrared_bars.png",
//    @"barcodes3.png",
//    @"barcodes.jpg",
//    @"qrcode.png",
    @"binarized1.jpg",
#endif
    ];
    g_viewController = self;
#ifdef GRAPH_PROCESS_LAB
    [self.tableView.superview bringSubviewToFront:self.tableView];
    self.tableView.contentInset = UIEdgeInsetsMake(60, 0, 0, 0);
#else
    self.tableView.hidden = YES;
#endif
    for (NSString* fileName in sourceFileNames)
    {
        NSString* imagePath = [[NSBundle mainBundle] pathForResource:fileName ofType:nil];
#ifdef GRAPH_PROCESS_LAB
        [self processImageV1:imagePath];
#else
        [self preprocessImage:imagePath];
#endif
    }
}

-(void) prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender {
    if ([segue.identifier isEqualToString:@"toResults"])
    {
        ScanResultsTableViewController* vc = segue.destinationViewController;
        vc.barcodeImages = self.barcodeImages;
        vc.barcodeNames = self.barcodeNames;
    }
}

-(IBAction) onRatioSliderValueChanged:(id)sender {
    /*
    float ratio = _ratioSlider.value;
#ifdef TEST_ZBAR
    float theta = _lineNormalTheta + TEST_ZBAR_THETA_OFFSET;///!!!For Debug
#else
    float theta = _lineNormalTheta;
#endif
    Mat barImage = rasterImageWithDirectionOneLine(_sourceCVImage, theta, ratio);
//    Mat barImage = _sourceCVImage;///!!!For Debug
    _dstImageView.image = UIImageFromCVMat(barImage);
    
    ImageScanner scanner;
    scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);
    int width = barImage.cols;
    int height = barImage.rows;
    uchar* raw = (uchar*) barImage.data;
    Image imageZbar(width, height, "Y800", raw, width * height);
    scanner.scan(imageZbar); //扫描条码
    Image::SymbolIterator symbol = imageZbar.symbol_begin();
    if(imageZbar.symbol_begin()==imageZbar.symbol_end())
    {
//        cout<<"查询条码失败，请检查图片！"<<endl;
        _logLabel.text = @"查询条码失败";
    }
    for(;symbol != imageZbar.symbol_end();++symbol)
    {
//        cout<<"类型："<<endl<<symbol->get_type_name()<<endl<<endl;
//        cout<<"条码："<<endl<<symbol->get_data()<<endl<<endl;
        _logLabel.text = [NSString stringWithFormat:@"类型='%s' 条码='%s'", symbol->get_type_name().c_str(), symbol->get_data().c_str()];
    }
    imageZbar.set_data(NULL, 0);
    
    Mat srcImageWithLine = drawImageWithHoughLine(_sourceCVImage, theta, ratio);
    _srcImageView.image = UIImageFromCVMat(srcImageWithLine);
    /*/
    [self updatePreprocessedImage];
    //*/
}

-(IBAction) onBinaryThresholdSliderValueChanged:(id)sender {
    [self updatePreprocessedImage];
}

-(IBAction) onMedianBlurSizeSliderValueChanged:(id)sender {
    [self updatePreprocessedImage];
}

-(IBAction) onThetaSliderValueChanged:(id)sender {
    [self updatePreprocessedImage];
}

-(IBAction) onScanButtonClicked:(id)sender {
    self.logLabel.text = @"Scanning...";
    [self autoScanImage];
//    int medianBlurSize = roundf(_medianBlurSizeSlider.value);
//    if (medianBlurSize % 2 == 0)
//    {
//        medianBlurSize++;
//    }
//    float threshold = _binaryThresholdSlider.value;
//    float theta = _thetaSlider.value * M_PI / 180.f;
//    Mat& srcImg = _sourceCVImage;
//    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
//        [self scanBarcodesInImage:srcImg thresholdValue:threshold medianBlurSize:medianBlurSize theta:theta];
//    });
}

-(IBAction) onSelectImageButtonClicked:(id)sender {
    UIImagePickerController* imagePickerController = [[UIImagePickerController alloc] init];
    imagePickerController.delegate = self;
    imagePickerController.allowsEditing = NO;
    imagePickerController.sourceType = UIImagePickerControllerSourceTypeSavedPhotosAlbum;
    [self presentViewController:imagePickerController animated:YES completion:nil];
}

-(void) imagePickerControllerDidCancel:(UIImagePickerController *)picker
{
    [self dismissViewControllerAnimated:YES completion:nil];
}

-(void) imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<UIImagePickerControllerInfoKey, id> *)info {
    [picker dismissViewControllerAnimated:YES completion:nil];
    NSURL* imagePathURL = [info objectForKey:UIImagePickerControllerImageURL];
    NSString* imagePath = [imagePathURL.absoluteString substringFromIndex:@"file://".length];
#ifdef GRAPH_PROCESS_LAB
        [self processImageV1:imagePath];
#else
    [self preprocessImage:imagePath];
#endif
}

-(UITableViewCell*) tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    ImageCell* cell = (ImageCell
                        *) [tableView dequeueReusableCellWithIdentifier:@"ImageCell"];
    cell.imageView.image = _images[indexPath.row];
    cell.titleLabel.text = _imageTitles[indexPath.row];
    return cell;
}

-(NSInteger) tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section {
    return _images.count;
}

-(CGFloat) tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)indexPath {
    UIImage* img = _images[indexPath.row];
    return img.size.height + 36;
}

-(void) scrollViewDidEndDragging:(UIScrollView *)scrollView willDecelerate:(BOOL)decelerate {
    if (self.tableView.contentOffset.y < -20.f)
    {
        [self onSelectImageButtonClicked:nil];
    }
}

@end
