// #include <iostream>
// #include <fstream>
// #include <string>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <filesystem>  // C++17
// #include <numeric>     // std::accumulate

// // 这两个头文件请确保在编译命令中 -I 指向上一层include目录
// #include "../include/mmio.h"
// #include "../include/csr2csc.h"

// namespace fs = std::filesystem;

// /**
//  * 计算 Gini 系数的辅助函数
//  * 参数 arr: 存储每个行(或列)非零元数量的向量
//  * 返回 Gini 系数
//  */
// double computeGini(const std::vector<long long> &arr)
// {
//     if (arr.empty()) {
//         std::cerr << "[Warning] Input array is empty. Returning Gini = 0." << std::endl;
//         return 0.0;
//     }

//     std::vector<long long> sortedArr = arr;
//     std::sort(sortedArr.begin(), sortedArr.end());

//     long double sumVal = std::accumulate(sortedArr.begin(), sortedArr.end(), 0.0L);
//     if (sumVal == 0) {
//         std::cerr << "[Warning] Total non-zero entries sum is zero. Returning Gini = 0." << std::endl;
//         return 0.0;
//     }

//     long long n = (long long)sortedArr.size();
//     long double cum = 0.0;
//     for (long long i = 0; i < n; ++i) {
//         cum += sortedArr[i] * (i + 1);
//     }

//     long double numerator = (long double)(n + 1) * sumVal - 2.0L * cum;
//     long double denominator = (long double)n * sumVal;
//     long double G = numerator / denominator;

//     return std::max(0.0L, std::min(1.0L, G));
// }

// /**
//  * 计算给定分布向量的统计量。
//  */
// struct DistStats {
//     double meanVal;
//     double stdVal;
//     long long minVal;
//     long long maxVal;
//     double gini;
//     std::vector<double> hotCoverage;
//     double pRatio;
//     std::vector<double> topRatios;
// };

// DistStats computeDistStats(const std::vector<long long> &arr, const std::vector<double> &coverageRatios)
// {
//     DistStats stats;
//     if (arr.empty()) {
//         stats.meanVal = 0.0; stats.stdVal = 0.0;
//         stats.minVal = 0;    stats.maxVal = 0;
//         stats.gini = 0.0;
//         stats.hotCoverage.assign(coverageRatios.size(), 0.0);
//         stats.pRatio = 0.0;
//         stats.topRatios.assign(8, 0.0); // 更新为8个值
//         return stats;
//     }

//     long long n = (long long)arr.size();
//     long long sumVal = std::accumulate(arr.begin(), arr.end(), 0LL);
//     double meanVal = (double)sumVal / n;

//     // 计算 std
//     long double varSum = 0.0L;
//     for (auto &v : arr) {
//         long double diff = (long double)v - meanVal;
//         varSum += diff * diff;
//     }
//     double variance = (double)(varSum / n);
//     double stdVal = std::sqrt(variance);

//     // 计算 gini
//     double gini = computeGini(arr);

//     // 计算 p-ratio
//     long long maxVal = *std::max_element(arr.begin(), arr.end());
//     long long minVal = *std::min_element(arr.begin(), arr.end());
//     double pRatio = (minVal > 0) ? (double)maxVal / minVal : 0.0;

//     // 计算 top k ratios
//     std::vector<long long> sortedArr = arr;
//     std::sort(sortedArr.begin(), sortedArr.end(), std::greater<long long>());
//     std::vector<int> topKs = {1, 2, 3, 4, 5, 10, 50, 100, 200, 300}; // 新增topKs范围
//     stats.topRatios.resize(topKs.size(), 0.0);
//     for (size_t i = 0; i < topKs.size(); i++) {
//         int k = topKs[i];
//         if (k - 1 < (int)sortedArr.size()) {
//             stats.topRatios[i] = meanVal > 0 ? (double)sortedArr[k - 1] / meanVal : 0.0;
//         }
//     }

//     // 计算热点覆盖率
//     double totalNNZ = (double)sumVal;
//     stats.hotCoverage.resize(coverageRatios.size(), 0.0);
//     for (int c = 0; c < (int)coverageRatios.size(); c++) {
//         double targetCoverage = coverageRatios[c];
//         double cumNNZ = 0.0;
//         for (int i = 0; i < (int)sortedArr.size(); i++) {
//             cumNNZ += sortedArr[i];
//             if (cumNNZ / totalNNZ >= targetCoverage) {
//                 stats.hotCoverage[c] = (double)(i + 1) / n; // 用于覆盖目标的行/列比例
//                 break;
//             }
//         }
//     }

//     stats.meanVal = meanVal;
//     stats.stdVal = stdVal;
//     stats.minVal = minVal;
//     stats.maxVal = maxVal;
//     stats.gini = gini;
//     stats.pRatio = pRatio;

//     return stats;
// }

// int main(int argc, char* argv[])
// {
//     if (argc < 3) {
//         std::cerr << "Usage: " << argv[0] 
//                   << " <mtx_root_folder> <output_csv>\n";
//         return 1;
//     }

//     std::string mtxRoot = argv[1];
//     std::string outCsv  = argv[2];

//     std::ofstream ofs(outCsv);
//     if (!ofs.is_open()) {
//         std::cerr << "[Error] Cannot open output CSV: " << outCsv << "\n";
//         return 1;
//     }

//     ofs << "MatrixName,nrows,ncols,nnz,rowMean,rowStd,rowMin,rowMax,rowGini,rowPRatio,";
//     ofs << "rowTop1,rowTop2,rowTop3,rowTop4,rowTop5,rowTop10,rowTop50,rowTop100,rowTop200,rowTop300,";
//     ofs << "rowHot10,rowHot20,rowHot30,rowHot40,rowHot50,rowHot60,rowHot70,rowHot80,rowHot90,";
//     ofs << "colMean,colStd,colMin,colMax,colGini,colPRatio,";
//     ofs << "colTop1,colTop2,colTop3,colTop4,colTop5,colTop10,colTop50,colTop100,colTop200,colTop300,";
//     ofs << "colHot10,colHot20,colHot30,colHot40,colHot50,colHot60,colHot70,colHot80,colHot90\n";

//     std::vector<double> coverageRatios = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

//     for (const auto & entry : fs::directory_iterator(mtxRoot)) {
//         if (!entry.is_directory()) {
//             continue;
//         }

//         std::string matrixName = entry.path().filename().string();
//         fs::path mtxFile = entry.path() / (matrixName + ".mtx");
//         if (!fs::exists(mtxFile)) {
//             continue;
//         }

//         int rowA, colA;
//         indT nnzA;
//         int isSymmetricA;
//         valT *csrVal = nullptr;
//         int *csrColInd = nullptr;
//         int *csrRowPtr = nullptr;

//         char * filenameCStr = new char[mtxFile.string().size() + 1];
//         std::strcpy(filenameCStr, mtxFile.string().c_str());
//         mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA,
//                       &csrRowPtr, &csrColInd, &csrVal,
//                       filenameCStr);
//         delete[] filenameCStr;

//         valT *cscVal = nullptr;
//         int *cscRowInd = nullptr;
//         int *cscColPtr = nullptr;
//         csr2csc(csrVal, csrRowPtr, csrColInd, rowA, colA, nnzA,
//                 &cscVal, &cscColPtr, &cscRowInd);

//         long long nrows = rowA;
//         long long ncols = colA;
//         long long nnz = nnzA;

//         std::vector<long long> rowNNZ(nrows, 0LL);
//         for (int i = 0; i < nrows; i++) {
//             long long start = csrRowPtr[i];
//             long long end = csrRowPtr[i + 1];
//             rowNNZ[i] = (end - start);
//         }
//         DistStats rowStats = computeDistStats(rowNNZ, coverageRatios);

//         std::vector<long long> colNNZ(ncols, 0LL);
//         for (int j = 0; j < ncols; j++) {
//             long long start = cscColPtr[j];
//             long long end = cscColPtr[j + 1];
//             colNNZ[j] = (end - start);
//         }
//         DistStats colStats = computeDistStats(colNNZ, coverageRatios);

//         ofs << matrixName << "," << nrows << "," << ncols << "," << nnz << ","
//             << rowStats.meanVal << "," << rowStats.stdVal << "," << rowStats.minVal << ","
//             << rowStats.maxVal << "," << rowStats.gini << "," << rowStats.pRatio << ",";

//         for (double ratio : rowStats.topRatios) {
//             ofs << ratio << ",";
//         }
//         for (size_t i = 0; i < rowStats.hotCoverage.size(); i++) {
//             ofs << rowStats.hotCoverage[i];
//             if (i + 1 < rowStats.hotCoverage.size()) ofs << ",";
//         }
//         ofs << ",";

//         ofs << colStats.meanVal << "," << colStats.stdVal << "," << colStats.minVal << ","
//             << colStats.maxVal << "," << colStats.gini << "," << colStats.pRatio << ",";

//         for (double ratio : colStats.topRatios) {
//             ofs << ratio << ",";
//         }
//         for (size_t i = 0; i < colStats.hotCoverage.size(); i++) {
//             ofs << colStats.hotCoverage[i];
//             if (i + 1 < colStats.hotCoverage.size()) ofs << ",";
//         }
//         ofs << "\n";

//         delete[] csrVal;
//         delete[] csrColInd;
//         delete[] csrRowPtr;
//         delete[] cscVal;
//         delete[] cscRowInd;
//         delete[] cscColPtr;
//     }

//     ofs.close();
//     std::cout << "Feature extraction done. Results saved to " << outCsv << std::endl;
//     return 0;
// }

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <filesystem>  // C++17
#include <numeric>     // std::accumulate

// 这两个头文件请确保在编译命令中 -I 指向上一层include目录
#include "../include/mmio.h"
#include "../include/csr2csc.h"

namespace fs = std::filesystem;

/**
 * 计算给定分布向量的统计量。
 */
struct DistStats {
    double meanVal;
    double stdVal;
    long long minVal;
    long long maxVal;
    std::vector<double> hotCoverage;
    double pRatio;
    std::vector<double> topRatios;
};

DistStats computeDistStats(const std::vector<long long> &arr, const std::vector<double> &coverageRatios)
{
    DistStats stats;
    if (arr.empty()) {
        stats.meanVal = 0.0; stats.stdVal = 0.0;
        stats.minVal = 0;    stats.maxVal = 0;
        stats.hotCoverage.assign(coverageRatios.size(), 0.0);
        stats.pRatio = 0.0;
        stats.topRatios.assign(8, 0.0); // 更新为8个值
        return stats;
    }

    long long n = (long long)arr.size();
    long long sumVal = std::accumulate(arr.begin(), arr.end(), 0LL);
    double meanVal = (double)sumVal / n;

    // 计算 std
    long double varSum = 0.0L;
    for (auto &v : arr) {
        long double diff = (long double)v - meanVal;
        varSum += diff * diff;
    }
    double variance = (double)(varSum / n);
    double stdVal = std::sqrt(variance);

    // 计算 p-ratio
    long long maxVal = *std::max_element(arr.begin(), arr.end());
    long long minVal = *std::min_element(arr.begin(), arr.end());
    double pRatio = (minVal > 0) ? (double)maxVal / minVal : 0.0;

    // 计算 top k ratios
    std::vector<long long> sortedArr = arr;
    std::sort(sortedArr.begin(), sortedArr.end(), std::greater<long long>());
    std::vector<int> topKs = {1, 2, 3, 4, 5, 10, 50, 100, 200, 300}; // 新增topKs范围
    stats.topRatios.resize(topKs.size(), 0.0);
    for (size_t i = 0; i < topKs.size(); i++) {
        int k = topKs[i];
        if (k - 1 < (int)sortedArr.size()) {
            stats.topRatios[i] = meanVal > 0 ? (double)sortedArr[k - 1] / meanVal : 0.0;
        }
    }

    // 计算热点覆盖率
    double totalNNZ = (double)sumVal;
    stats.hotCoverage.resize(coverageRatios.size(), 0.0);
    for (int c = 0; c < (int)coverageRatios.size(); c++) {
        double targetCoverage = coverageRatios[c];
        double cumNNZ = 0.0;
        for (int i = 0; i < (int)sortedArr.size(); i++) {
            cumNNZ += sortedArr[i];
            if (cumNNZ / totalNNZ >= targetCoverage) {
                stats.hotCoverage[c] = (double)(i + 1) / n; // 用于覆盖目标的行/列比例
                break;
            }
        }
    }

    stats.meanVal = meanVal;
    stats.stdVal = stdVal;
    stats.minVal = minVal;
    stats.maxVal = maxVal;
    stats.pRatio = pRatio;

    return stats;
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <mtx_root_folder> <output_csv>\n";
        return 1;
    }

    std::string mtxRoot = argv[1];
    std::string outCsv  = argv[2];

    std::ofstream ofs(outCsv);
    if (!ofs.is_open()) {
        std::cerr << "[Error] Cannot open output CSV: " << outCsv << "\n";
        return 1;
    }

    ofs << "MatrixName,nrows,ncols,nnz,rowMean,rowStd,rowMin,rowMax,rowPRatio,";
    ofs << "rowTop1,rowTop2,rowTop3,rowTop4,rowTop5,rowTop10,rowTop50,rowTop100,rowTop200,rowTop300,";
    ofs << "rowHot10,rowHot20,rowHot30,rowHot40,rowHot50,rowHot60,rowHot70,rowHot80,rowHot90,";
    ofs << "colMean,colStd,colMin,colMax,colPRatio,";
    ofs << "colTop1,colTop2,colTop3,colTop4,colTop5,colTop10,colTop50,colTop100,colTop200,colTop300,";
    ofs << "colHot10,colHot20,colHot30,colHot40,colHot50,colHot60,colHot70,colHot80,colHot90\n";

    std::vector<double> coverageRatios = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    for (const auto & entry : fs::directory_iterator(mtxRoot)) {
        if (!entry.is_directory()) {
            continue;
        }

        std::string matrixName = entry.path().filename().string();
        fs::path mtxFile = entry.path() / (matrixName + ".mtx");
        if (!fs::exists(mtxFile)) {
            continue;
        }

        int rowA, colA;
        indT nnzA;
        int isSymmetricA;
        valT *csrVal = nullptr;
        int *csrColInd = nullptr;
        int *csrRowPtr = nullptr;

        char * filenameCStr = new char[mtxFile.string().size() + 1];
        std::strcpy(filenameCStr, mtxFile.string().c_str());
        mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA,
                      &csrRowPtr, &csrColInd, &csrVal,
                      filenameCStr);
        delete[] filenameCStr;

        valT *cscVal = nullptr;
        int *cscRowInd = nullptr;
        int *cscColPtr = nullptr;
        csr2csc(csrVal, csrRowPtr, csrColInd, rowA, colA, nnzA,
                &cscVal, &cscColPtr, &cscRowInd);

        long long nrows = rowA;
        long long ncols = colA;
        long long nnz = nnzA;

        std::vector<long long> rowNNZ(nrows, 0LL);
        for (int i = 0; i < nrows; i++) {
            long long start = csrRowPtr[i];
            long long end = csrRowPtr[i + 1];
            rowNNZ[i] = (end - start);
        }
        DistStats rowStats = computeDistStats(rowNNZ, coverageRatios);

        std::vector<long long> colNNZ(ncols, 0LL);
        for (int j = 0; j < ncols; j++) {
            long long start = cscColPtr[j];
            long long end = cscColPtr[j + 1];
            colNNZ[j] = (end - start);
        }
        DistStats colStats = computeDistStats(colNNZ, coverageRatios);

        ofs << matrixName << "," << nrows << "," << ncols << "," << nnz << ","
            << rowStats.meanVal << "," << rowStats.stdVal << "," << rowStats.minVal << ","
            << rowStats.maxVal << "," << rowStats.pRatio << ",";

        for (double ratio : rowStats.topRatios) {
            ofs << ratio << ",";
        }
        for (size_t i = 0; i < rowStats.hotCoverage.size(); i++) {
            ofs << rowStats.hotCoverage[i];
            if (i + 1 < rowStats.hotCoverage.size()) ofs << ",";
        }
        ofs << ",";

        ofs << colStats.meanVal << "," << colStats.stdVal << "," << colStats.minVal << ","
            << colStats.maxVal << "," << colStats.pRatio << ",";

        for (double ratio : colStats.topRatios) {
            ofs << ratio << ",";
        }
        for (size_t i = 0; i < colStats.hotCoverage.size(); i++) {
            ofs << colStats.hotCoverage[i];
            if (i + 1 < colStats.hotCoverage.size()) ofs << ",";
        }
        ofs << "\n";

        delete[] csrVal;
        delete[] csrColInd;
        delete[] csrRowPtr;
        delete[] cscVal;
        delete[] cscRowInd;
        delete[] cscColPtr;
    }

    ofs.close();
    std::cout << "Feature extraction done. Results saved to " << outCsv << std::endl;
    return 0;
}
