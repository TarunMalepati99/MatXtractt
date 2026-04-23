$file = "C:\Users\tarun\MatXtract\ML\load_and_predict_and_run_spmv.py"
$content = Get-Content $file -Raw

$oldFunc = 'def execute_tcspmv_test(matrix_name, col_frac, hot_frac):
    """
    Execute the specified command line program ../build/matxtract_perftest.
    """
    matrix_path = f"../../../data/mtx/{matrix_name}/{matrix_name}.mtx"
    cmd = [
        "../build/matxtract_perftest",
        f"{col_frac}",
        f"{hot_frac}",
        matrix_path
    ]
    try:
        output = subprocess.check_output(cmd, universal_newlines=True)
        print(f"[Info] Command executed successfully for {matrix_name}")
        print(output)
    except subprocess.CalledProcessError as e:
        print(f"[Error] Command failed for {matrix_name}: {e}")'

$newFunc = 'def execute_tcspmv_test(matrix_name, col_frac, hot_frac):
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    build_dir   = os.path.join(script_dir, "..", "build")
    data_dir    = os.path.join(script_dir, "..", "data", "mtx")
    matrix_path = os.path.join(data_dir, matrix_name, f"{matrix_name}.mtx")
    exe_path    = os.path.join(build_dir, "matxtract_perftest.exe")
    cmd = [exe_path, f"{col_frac}", f"{hot_frac}", matrix_path]
    try:
        output = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT)
        print(f"[Info] Command executed successfully for {matrix_name}")
        print(output)
        return output
    except subprocess.CalledProcessError as e:
        print(f"[Error] Command failed for {matrix_name}: {e}")
        print(e.output)
        return None'

$content = $content.Replace($oldFunc, $newFunc)
Set-Content $file $content -Encoding utf8
Write-Host "File patched. Testing..."
& "C:\Users\tarun\AppData\Local\Python\bin\python.exe" "C:\Users\tarun\MatXtract\ML\load_and_predict_and_run_spmv.py"