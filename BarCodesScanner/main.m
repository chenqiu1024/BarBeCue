//
//  main.m
//  BarCodesScanner
//
//  Created by qiudong on 2019/11/25.
//  Copyright © 2019 qiudong. All rights reserved.
//

//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <iostream>
//#include "zbar.h"
//#import <UIKit/UIKit.h>
//#import <CoreGraphics/CoreGraphics.h>
#import "AppDelegate.h"

//using namespace std;
////using namespace zbar;
//using namespace cv;
//
//cv::Mat cvMatFromUIImage(UIImage* image)
//{
//  CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
//  CGFloat cols = image.size.width;
//  CGFloat rows = image.size.height;
//
//  cv::Mat cvMat(rows, cols, CV_8UC4);// 8 bits per component, 4 channels (color channels + alpha)
//
//  CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
//                                                 cols,                       // Width of bitmap
//                                                 rows,                       // Height of bitmap
//                                                 8,                          // Bits per component
//                                                 cvMat.step[0],              // Bytes per row
//                                                 colorSpace,                 // Colorspace
//                                                 kCGImageAlphaNoneSkipLast |
//                                                 kCGBitmapByteOrderDefault); // Bitmap info flags
//
//  CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
//  CGContextRelease(contextRef);
//
//  return cvMat;
//}
//
//cv::Mat cvMatGrayFromUIImage(UIImage* image)
//{
//  CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
//  CGFloat cols = image.size.width;
//  CGFloat rows = image.size.height;
//
//  cv::Mat cvMat(rows, cols, CV_8UC1); // 8 bits per component, 1 channels
//
//  CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to data
//                                                 cols,                       // Width of bitmap
//                                                 rows,                       // Height of bitmap
//                                                 8,                          // Bits per component
//                                                 cvMat.step[0],              // Bytes per row
//                                                 colorSpace,                 // Colorspace
//                                                 kCGImageAlphaNoneSkipLast |
//                                                 kCGBitmapByteOrderDefault); // Bitmap info flags
//
//  CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
//  CGContextRelease(contextRef);
//
//  return cvMat;
//}
////After the processing we need to convert it back to UIImage. The code below can handle both gray-scale and color image conversions (determined by the number of channels in the if statement).
////
////cv::Mat greyMat;
////cv::cvtColor(inputMat, greyMat, CV_BGR2GRAY);
//UIImage* UIImageFromCVMat(cv::Mat cvMat)
//{
//  NSData *data = [NSData dataWithBytes:cvMat.data length:cvMat.elemSize()*cvMat.total()];
//  CGColorSpaceRef colorSpace;
//
//  if (cvMat.elemSize() == 1) {
//      colorSpace = CGColorSpaceCreateDeviceGray();
//  } else {
//      colorSpace = CGColorSpaceCreateDeviceRGB();
//  }
//
//  CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
//
//  // Creating CGImage from cv::Mat
//  CGImageRef imageRef = CGImageCreate(cvMat.cols,                                 //width
//                                     cvMat.rows,                                 //height
//                                     8,                                          //bits per component
//                                     8 * cvMat.elemSize(),                       //bits per pixel
//                                     cvMat.step[0],                            //bytesPerRow
//                                     colorSpace,                                 //colorspace
//                                     kCGImageAlphaNone|kCGBitmapByteOrderDefault,// bitmap info
//                                     provider,                                   //CGDataProviderRef
//                                     NULL,                                       //decode
//                                     false,                                      //should interpolate
//                                     kCGRenderingIntentDefault                   //intent
//                                     );
//
//
//  // Getting UIImage from CGImage
//  UIImage *finalImage = [UIImage imageWithCGImage:imageRef];
//  CGImageRelease(imageRef);
//  CGDataProviderRelease(provider);
//  CGColorSpaceRelease(colorSpace);
//
//  return finalImage;
//}
//
//void showImage(Mat& mat, NSString* title) {
////    imshow(title.UTF8String, image);
//
//    UIImage* img = UIImageFromCVMat(mat);
//    NSString* outPath = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0] stringByAppendingPathComponent:[NSString stringWithFormat:@"%@.png", title]];
//    NSData* pngData = UIImagePNGRepresentation(img);
//    [pngData writeToFile:outPath
//              atomically:NO];
//}
//#include "cyclic.h"
//#include "kMeansCluster.h"

int main(int argc,char *argv[])
{
//    cyclic_decoder_t* decoder = (cyclic_decoder_t*) malloc(sizeof(cyclic_decoder_t));
//    cyclic_reset(decoder);
//    cyclic_destroy(decoder);
//    kMeansCluster_test();

//    NSString* docPath = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0];
//    NSFileManager* fm = [NSFileManager defaultManager];
//    NSString* destDirectoy = [docPath stringByAppendingPathComponent:@"output"];
//    [fm createDirectoryAtPath:destDirectoy withIntermediateDirectories:YES attributes:nil error:nil];
//    NSDirectoryEnumerator<NSString* >* iter = [fm enumeratorAtPath:docPath];
//    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
//    //文字居中显示在画布上
//    NSMutableParagraphStyle* paragraphStyle = [[NSParagraphStyle defaultParagraphStyle] mutableCopy];
//    paragraphStyle.lineBreakMode = NSLineBreakByCharWrapping;
//    paragraphStyle.alignment = NSTextAlignmentCenter;//文字居中
//    const CGFloat fontSize = 36.f;
//    NSShadow* shadow = [[NSShadow alloc] init];
//    shadow.shadowColor = [UIColor yellowColor];
//    shadow.shadowOffset = CGSizeMake(1.f, 1.f);
//    shadow.shadowBlurRadius = 1.f;
//    for (NSString* filename in iter)
//    {
//        if (![filename.pathExtension isEqualToString:@"png"]) continue;
//        NSLog(@"%@", filename);
//        NSString* filePath = [docPath stringByAppendingPathComponent:filename];
//        NSString* dstFilePath = [[destDirectoy stringByAppendingPathComponent:[filename stringByDeletingPathExtension]] stringByAppendingPathExtension:@"jpg"];
//        UIImage* srcImage = [UIImage imageWithContentsOfFile:filePath];
//        CGContextRef cgCtx = CGBitmapContextCreate(NULL, srcImage.size.width, srcImage.size.height, 8, 4 * srcImage.size.width, colorSpace, kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
//        CGContextScaleCTM(cgCtx, 1.f, -1.f);
//        CGContextTranslateCTM(cgCtx, 0, -srcImage.size.height);
//        CGContextDrawImage(cgCtx, CGRectMake(0, 0, srcImage.size.width, srcImage.size.height), srcImage.CGImage);
//
//        UIGraphicsPushContext(cgCtx);
//        //计算文字所占的size,文字居中显示在画布上
//        CGRect textBoundRect = CGRectMake(srcImage.size.width * 0.125, srcImage.size.height * 0.75, srcImage.size.width * 0.75, srcImage.size.height * 0.25);
//        CGSize textSize = [filename boundingRectWithSize:textBoundRect.size options:NSStringDrawingUsesLineFragmentOrigin
//                                           attributes:@{NSFontAttributeName:[UIFont systemFontOfSize:fontSize]} context:nil].size;
//        CGRect textRect = CGRectMake(CGRectGetMidX(textBoundRect) - textSize.width / 2, CGRectGetMidY(textBoundRect) - textSize.height / 2, textSize.width, textSize.height);
//        [filename drawInRect:textRect withAttributes:@{
//            NSFontAttributeName:[UIFont systemFontOfSize:fontSize],
//            NSForegroundColorAttributeName:[UIColor redColor],
//            NSShadowAttributeName: shadow,
//            NSParagraphStyleAttributeName:paragraphStyle}];
////        void* data = CGBitmapContextGetData(cgCtx);
//        UIGraphicsPopContext();
//
//        CGImageRef cgImg = CGBitmapContextCreateImage(cgCtx);
//        UIImage* dstImage = [UIImage imageWithCGImage:cgImg];
//        NSData* dstData = UIImageJPEGRepresentation(dstImage, 0.9);
//        [dstData writeToFile:dstFilePath atomically:NO];
//        CGImageRelease(cgImg);
//        CGContextRelease(cgCtx);
//    }
//    CGColorSpaceRelease(colorSpace);
    
    NSString * appDelegateClassName;
    @autoreleasepool {
        // Setup code that might create autoreleased objects goes here.
        appDelegateClassName = NSStringFromClass([AppDelegate class]);
    }
    return UIApplicationMain(argc, argv, nil, appDelegateClassName);
}

//int main(int argc, char * argv[]) {
    
//}
