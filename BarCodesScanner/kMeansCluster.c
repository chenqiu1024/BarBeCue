//
//  kMeansCluster.c
//  BarCodesScanner
//
//  Created by DOM QIU on 2019/12/22.
//  Copyright © 2019 qiudong. All rights reserved.
//

#include "kMeansCluster.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

//#define TEST_K_MEANS_CLUSTERING

float SquareEuclideanDistance(const float* v0, const float* v1, int elements) {
    float sum = 0.f;
    for (int i = elements - 1; i >= 0; --i)
    {
        float d = v1[i] - v0[i];
        sum += d * d;
    }
    return sum;
}

void printArray(int* a, int length) {
    for (int p=0; p<length; ++p) printf("%d, ", a[p]);
    printf("\n");
}

int* kMeansCluster(const float** datas, int elements, int dataCount, int k) {
    if (!datas || elements < 1 || dataCount < 1 || k < 1) return NULL;

    int* ret = (int*) malloc(sizeof(int) * dataCount);
    for (int i = dataCount - 1; i >= 0; --i)
    {
        ret[i] = i;
    }

    if (k >= dataCount) return ret;
    
//    unsigned int seed = *((unsigned int*)datas[0]);
//    srand(seed);
//    printf("#kMeans# seed=%d\n", seed);
    for (int i = 0; i < k; ++i)
    {
        int j = i + (rand() % (dataCount - i));
//        printf("#kMeans# j=%d\n", j);
        if (j != i)
        {
            ret[j] ^= ret[i];
            ret[i] ^= ret[j];
            ret[j] ^= ret[i];
        }
    }
    for (int i = k; i < dataCount; ++i) ret[i] = -1;
//    printArray(ret, dataCount);
    for (int i = k - 1; i >= 0; --i)
    {
        int j = 0;
        for (; j < k; ++j)
        {
            if (0 > ret[j]) continue;
            if (-1 == ret[ret[j]])
            {
                ret[ret[j]] = -2 - j;
                ret[j] = -1;
                break;
            }
        }
//        printArray(ret, dataCount);
        if (j == k)
        {
            for (j = 0; j < dataCount; ++j)
            {
                if (ret[j] >= 0)
                {
                    ret[j] = -2 - j;
                }
            }
            break;
        }
    }
#ifndef TEST_K_MEANS_CLUSTERING
    float** centers = (float**) malloc(sizeof(float*) * k);
    memset(centers, 0, sizeof(float*) * k);///!!!For Debug
    for (int i = dataCount - 1; i >= 0; --i)
    {
        int n = -2 - ret[i];
        if (n >= 0)
        {
            centers[n] = (float*) malloc(sizeof(float) * elements);
            memcpy(centers[n], datas[i], sizeof(float) * elements);
        }
    }
//    for (int i = 0; i < k; ++i)
//    {
//        if (!centers[i])
//        {
//            centers[i] = NULL;///!!!For Debug
//        }
//    }
    for (int i = dataCount - 1; i >= 0; --i)
    {
        int n = -2 - ret[i];
        ret[i] = n;
    }

    const int MaxIteration = dataCount;
    int* clusterSize = (int*) malloc(sizeof(int) * k);
    for (int iIter = MaxIteration - 1 ; iIter >= 0; --iIter)
    {
        bool noChange = true;
        for (int i = dataCount - 1; i >= 0; --i)
        {
            float minDist = SquareEuclideanDistance(datas[i], centers[0], elements);
            int minDistIndex = 0;
            for (int j = 1; j < k; ++j)
            {
                float dist = SquareEuclideanDistance(datas[i], centers[j], elements);
                if (dist < minDist)
                {
                    minDist = dist;
                    minDistIndex = j;
                }
            }
            if (minDistIndex != ret[i])
            {
                ret[i] = minDistIndex;
                noChange = false;
            }
        }
        if (noChange) break;
        
        for (int i = k - 1; i >= 0; --i)
        {
            for (int j = elements - 1; j >= 0; --j)
                centers[i][j] = 0.f;
        }
        memset(clusterSize, 0, sizeof(int) * k);
        for (int i = dataCount - 1; i >= 0; --i)
        {
            int cluster = ret[i];
            clusterSize[cluster]++;
            for (int j = elements - 1; j >= 0; --j)
            {
                centers[cluster][j] += datas[i][j];
            }
        }
        for (int i = k - 1; i >= 0; --i)
        {
            int n = clusterSize[i];
            for (int j = elements - 1; j >= 0; --j)
            {
                centers[i][j] /= n;
            }
        }
    }
    
    free(clusterSize);
    for (int i = k - 1; i >= 0; --i)
    {
        free(centers[i]);
    }
    free(centers);
#endif //#ifndef TEST_K_MEANS_CLUSTERING
    return ret;
}

void kMeansCluster_test(void) {
    float rawDatas[] = {
        1.51843643,
        1.55334306,
        1.51843643,
        1.55334306,
        1.53588974,
        1.53588974,
    };
    float** datas = (float**) malloc(sizeof(float*) * sizeof(rawDatas) / sizeof(rawDatas[0]));
    for (int i = sizeof(rawDatas) / sizeof(rawDatas[0]) - 1; i >= 0; --i)
    {
        datas[i] = (float*) malloc(sizeof(float));
        datas[i][0] = rawDatas[i];
    }

    int elements = 1;
    int dataCount = 6;
    int k = 4;
    
    int* initCenters = NULL;
    int* occurs = NULL;
    int seed;
    for (int i=0; i<1024; ++i)
    {
        seed = rand();
//        seed = 1069702176;
        srand(seed);
        
        if (k >= dataCount) continue;

        initCenters = kMeansCluster(datas, elements, dataCount, k);
        
        bool* occurs = (bool*) malloc(sizeof(bool) * k);
        memset(occurs, 0, sizeof(bool) * k);
        int nonOccupied = k;
        for (int j = 0; j < dataCount; ++j)
        {
            int n = -2 - initCenters[j];
            if (n >= k || n < -1)
                goto fail;
            else if (n >= 0)
            {
                if (!occurs[n])
                {
                    occurs[n] = true;
                    nonOccupied--;
                }
                else
                    goto fail;
            }
        }
        if (0 != nonOccupied)
            goto fail;

        printf("\nPassed: ");
        printArray(initCenters, dataCount);
        free(occurs);
        free(initCenters);
    }

    for (int i = sizeof(rawDatas) / sizeof(rawDatas[0]) - 1; i >= 0; --i)
    {
        free(datas[i]);
    }
    free(datas);
    return;
    
fail:
    printf("\nFailed: count=%d, k=%d, Seed=%d\n", dataCount, k, seed);
    printArray(initCenters, dataCount);

    free(occurs);
    free(initCenters);
    for (int i = sizeof(rawDatas) / sizeof(rawDatas[0]) - 1; i >= 0; --i)
    {
        free(datas[i]);
    }
    free(datas);

    exit(-1);
}
