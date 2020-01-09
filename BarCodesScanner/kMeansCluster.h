//
//  kMeansCluster.h
//  BarCodesScanner
//
//  Created by DOM QIU on 2019/12/22.
//  Copyright © 2019 qiudong. All rights reserved.
//

#ifndef kMeansCluster_h
#define kMeansCluster_h

#ifdef __cplusplus
extern "C" {
#endif

int* kMeansCluster(const float** datas, int elements, int dataCount, int k);

void kMeansCluster_test(void);

#ifdef __cplusplus
}
#endif

#endif /* kMeansCluster_h */
