//
//  ScanResultsTableViewController.h
//  BarCodesScanner
//
//  Created by qiudong on 2019/12/14.
//  Copyright © 2019 qiudong. All rights reserved.
//

#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface ScanResultsTableViewController : UITableViewController

@property (nonatomic, strong) NSMutableArray<UIImage* >* barcodeImages;
@property (nonatomic, strong) NSMutableArray<NSString* >* barcodeNames;

@end

NS_ASSUME_NONNULL_END
