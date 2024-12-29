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

// 说明：以下typedef/using与 mmio.h、csr2csc.h 中保持一致，
//       根据实际情况可能需要调整。

namespace fs = std::filesystem;

/**
 * 计算 Gini 系数的辅助函数
 * 参数 arr: 存储每个行(或列)非零元数量的向量
 * 返回 Gini 系数
 */
double computeGini(const std::vector<long long> &arr)
{
    if (arr.empty()) {
        std::cerr << "[Warning] Input array is empty. Returning Gini = 0." << std::endl;
        return 0.0;
    }

    std::vector<long long> sortedArr = arr;
    std::sort(sortedArr.begin(), sortedArr.end());

    long double sumVal = std::accumulate(sortedArr.begin(), sortedArr.end(), 0.0L);
    if (sumVal == 0) {
        std::cerr << "[Warning] Total non-zero entries sum is zero. Returning Gini = 0." << std::endl;
        return 0.0;
    }

    long long n = (long long)sortedArr.size();
    long double cum = 0.0;
    for (long long i = 0; i < n; ++i) {
        cum += sortedArr[i] * (i + 1);
    }

    long double numerator = (long double)(n + 1) * sumVal - 2.0L * cum;
    long double denominator = (long double)n * sumVal;
    long double G = numerator / denominator;

    return std::max(0.0L, std::min(1.0L, G));
}

/**
 * 计算给定分布向量(例如每行nnz数)，得到：
 * - mean, std, min, max, gini
 * 并计算“热点覆盖率(前10%,20%,...90%)”等百分比分布。
 *
 * 参数:
 *   arr: 每行(或每列)的 nnz 数量
 *   coverageRatios: 要计算的热点百分比，例如 {0.1,0.2,...,0.9} 表示前 10%, 20%, ... 90%
 * 返回:
 *   一个 struct 或者 vector<double> 存储各统计量 + 各热点覆盖值
 */
struct DistStats {
    double meanVal;
    double stdVal;
    long long minVal;
    long long maxVal;
    double gini;
    // 对于前10%, 20%, ..., 90%行的非零元占总非零元比例:
    std::vector<double> hotCoverage; 
};

DistStats computeDistStats(const std::vector<long long> &arr, const std::vector<double> &coverageRatios)
{
    DistStats stats;
    if (arr.empty()) {
        // 空数组情形: 全部置0
        stats.meanVal = 0.0; stats.stdVal = 0.0;
        stats.minVal = 0;    stats.maxVal = 0;
        stats.gini = 0.0;
        stats.hotCoverage.assign(coverageRatios.size(), 0.0);
        return stats;
    }

    long long n = (long long)arr.size();
    // 1) mean
    long long sumVal = 0;
    long long minv = LLONG_MAX;
    long long maxv = LLONG_MIN;
    for (auto &v : arr) {
        sumVal += v;
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
    }
    double meanVal = (double)sumVal / n;

    // 2) std
    long double varSum = 0.0L;
    for (auto &v : arr) {
        long double diff = (long double)v - meanVal;
        varSum += diff * diff;
    }
    double variance = (double)(varSum / n);
    double stdVal = std::sqrt(variance);

    // 3) gini
    double gini = computeGini(arr);

    // 4) min, max
    stats.meanVal = meanVal;
    stats.stdVal  = stdVal;
    stats.minVal  = minv;
    stats.maxVal  = maxv;
    stats.gini    = gini;

    // 5) 热点覆盖率
    //    "前10% 行(或列) 包含多少 % 的非零元？"
    //    通常做法：先按 nnz 降序排序，然后累加top K的nnz / 总nnz
    std::vector<long long> sortedArr = arr;
    std::sort(sortedArr.begin(), sortedArr.end(), std::greater<long long>());

    double totalNNZ = (double)sumVal;
    double cumNNZ = 0.0;
    int idx = 0;
    stats.hotCoverage.resize(coverageRatios.size(), 0.0);

    for (int i = 0; i < (int)sortedArr.size(); i++) {
        cumNNZ += sortedArr[i];
        // fraction of rows used so far
        double fracRowsUsed = (double)(i + 1) / (double)n;
        // coverage of nnz so far
        double coverageSoFar = cumNNZ / totalNNZ; // ratio in [0,1]

        // 对 coverageRatios 逐个比较
        // 不过这里我们想要的是 "前 X% 行 占总nnz的多少比例"
        //   => 当 fracRowsUsed >= coverageRatios[idx] 时，表示已到达前X%的行
        //   => coverageSoFar 即相应的 nnz 覆盖率
        // 也可反过来写: "top X% nnz 需要多少行" 等, 但此处按注释的做法实现
    }

    // 但是大多数情况下, '前10%行' 指的是 i < 0.1*n. 
    // 这里做法: for coverageRatios {0.1,0.2,...}, 取 top 0.1*n 行的nnz之和 / totalNNZ.
    // 因此更直接的是先计算 sortedArr 并定义 k = floor(ratio * n), 累加前k项
    for (int c = 0; c < (int)coverageRatios.size(); c++) {
        double ratio = coverageRatios[c]; // e.g. 0.1
        long long k = (long long)std::floor(ratio * n); 
        if (k < 1) k = 1;  // 避免0
        if (k > n) k = n;  // 安全保护

        long long partialSum = 0;
        for (long long i = 0; i < k; i++) {
            partialSum += sortedArr[i];
        }
        double coverage = (double)partialSum / totalNNZ;
        stats.hotCoverage[c] = coverage; 
    }

    return stats;
}

/**
 * 遍历 "/home/v-jiawcheng/wangluhan/data/mtx" 下所有子文件夹，
 * 每个子文件夹名字 = 矩阵名, 其中有 file.mtx => 读 CSR + CSC => 计算特征 => 写到 CSV
 */
int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <mtx_root_folder> <output_csv>\n";
        return 1;
    }

    std::string mtxRoot = argv[1];   // "/home/v-jiawcheng/wangluhan/data/mtx"
    std::string outCsv  = argv[2];   // "output.csv" 等

    // 打开输出 CSV
    std::ofstream ofs(outCsv);
    if (!ofs.is_open()) {
        std::cerr << "[Error] Cannot open output CSV: " << outCsv << "\n";
        return 1;
    }

    // 写表头(根据需求可增减)
    // 这里示例写了: name, nrows, ncols, nnz, rowMean, rowStd, rowMin, rowMax, rowGini, 
    //                rowHot10, rowHot20, ..., rowHot90,
    //                colMean, colStd, colMin, colMax, colGini,
    //                colHot10, colHot20, ..., colHot90
    ofs << "MatrixName,"
        << "nrows,"
        << "ncols,"
        << "nnz,"
        << "rowMean,"
        << "rowStd,"
        << "rowMin,"
        << "rowMax,"
        << "rowGini,"
        << "rowHot10,rowHot20,rowHot30,rowHot40,rowHot50,rowHot60,rowHot70,rowHot80,rowHot90,"
        << "colMean,"
        << "colStd,"
        << "colMin,"
        << "colMax,"
        << "colGini,"
        << "colHot10,colHot20,colHot30,colHot40,colHot50,colHot60,colHot70,colHot80,colHot90\n";

    // 定义热点比率(可根据需求微调)
    std::vector<double> coverageRatios = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    // 遍历根目录下的子文件夹
    for (const auto & entry : fs::directory_iterator(mtxRoot)) {
        if (!entry.is_directory()) {
            continue; // 仅处理文件夹
        }

        // 文件夹名(矩阵名)
        std::string matrixName = entry.path().filename().string(); 
        // 对应的 .mtx 文件名: <folder>/<folder>.mtx
        // 例如 /home/.../mtx/3Dspectralwave/3Dspectralwave.mtx
        fs::path mtxFile = entry.path() / (matrixName + ".mtx");
        if (!fs::exists(mtxFile)) {
            // 若不存在同名 mtx 文件, 跳过
            continue;
        }

        // ----------- 读取CSR ------------
        int rowA, colA;
        indT nnzA;
        int isSymmetricA;
        valT *csrVal   = nullptr;
        int  *csrColInd= nullptr;
        int  *csrRowPtr= nullptr;

        {
            char * filenameCStr = new char[mtxFile.string().size() + 1];
            std::strcpy(filenameCStr, mtxFile.string().c_str());
            mmio_allinone(&rowA, &colA, &nnzA, &isSymmetricA,
                          &csrRowPtr, &csrColInd, &csrVal,
                          filenameCStr);
            delete[] filenameCStr;
        }

        // ----------- 转换到 CSC ------------
        valT *cscVal   = nullptr;
        int  *cscRowInd= nullptr;
        int  *cscColPtr= nullptr;
        csr2csc(csrVal, csrRowPtr, csrColInd, rowA, colA, nnzA,
                &cscVal, &cscColPtr, &cscRowInd);

        // (A)Size: nrows, ncols, nnz
        long long nrows = rowA;
        long long ncols = colA;
        long long nnz   = nnzA;

        // (B) 构造 "每行 nnz 数" rowNNZ
        std::vector<long long> rowNNZ(nrows, 0LL);
        // csrRowPtr[i] 到 csrRowPtr[i+1]-1 是第 i 行的 nnz
        for(int i = 0; i < nrows; i++){
            long long start = csrRowPtr[i];
            long long end   = csrRowPtr[i+1];
            rowNNZ[i] = (end - start);
        }
        DistStats rowStats = computeDistStats(rowNNZ, coverageRatios);

        // (C) 构造 "每列 nnz 数" colNNZ
        std::vector<long long> colNNZ(ncols, 0LL);
        // cscColPtr[j] 到 cscColPtr[j+1]-1 是第 j 列的 nnz
        for(int j = 0; j < ncols; j++){
            long long start = cscColPtr[j];
            long long end   = cscColPtr[j+1];
            colNNZ[j] = (end - start);
        }
        DistStats colStats = computeDistStats(colNNZ, coverageRatios);

        // 将结果写到 CSV 中(一行)
        ofs << matrixName << ","
            << nrows << ","
            << ncols << ","
            << nnz << ","
            // row stats
            << rowStats.meanVal << ","
            << rowStats.stdVal << ","
            << rowStats.minVal << ","
            << rowStats.maxVal << ","
            << rowStats.gini << ",";

        // row hot coverage
        for (size_t i = 0; i < rowStats.hotCoverage.size(); i++) {
            ofs << rowStats.hotCoverage[i];
            if(i + 1 < rowStats.hotCoverage.size()) ofs << ",";
        }
        ofs << ",";

        // col stats
        ofs << colStats.meanVal << ","
            << colStats.stdVal << ","
            << colStats.minVal << ","
            << colStats.maxVal << ","
            << colStats.gini << ",";

        // col hot coverage
        for (size_t i = 0; i < colStats.hotCoverage.size(); i++) {
            ofs << colStats.hotCoverage[i];
            if(i + 1 < colStats.hotCoverage.size()) ofs << ",";
        }
        ofs << "\n";

        // 清理内存
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
