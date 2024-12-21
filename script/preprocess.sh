#!/bin/bash
cd "$(dirname "$0")"

# ../build/TCSpMVlib_tcperftest ../../../data/mtx/arabic-2005/arabic-2005.mtx 
# ../build/TCSpMVlib_tcperftest ../../../data/mtx/uk-2005/uk-2005.mtx
# ../build/TCSpMVlib_tcperftest ../../../data/mtx/uk-2002/uk-2002.mtx
# ../build/TCSpMVlib_tcperftest ../../../data/mtx/webbase-2001/webbase-2001.mtx
# ../build/TCSpMVlib_tcperftest ../../../data/mtx/sk-2005/sk-2005.mtx
# ../build/TCSpMVlib_tcperftest ../../../data/mtx/it-2004/it-2004.mtx
# ../build/TCSpMVlib_tcperftest ../../../data/large_mtx/GAP-web/GAP-web.mtx

../build/TCSpMVlib_perf ../../../data/mtx/arabic-2005/arabic-2005.mtx 
../build/TCSpMVlib_perf ../../../data/mtx/uk-2005/uk-2005.mtx
../build/TCSpMVlib_perf ../../../data/mtx/uk-2002/uk-2002.mtx
../build/TCSpMVlib_perf ../../../data/mtx/webbase-2001/webbase-2001.mtx
../build/TCSpMVlib_perf ../../../data/mtx/sk-2005/sk-2005.mtx
../build/TCSpMVlib_perf ../../../data/mtx/it-2004/it-2004.mtx
../build/TCSpMVlib_perf ../../../data/large_mtx/GAP-web/GAP-web.mtx

