# PYTHONPATH=$(pwd)/build-debug/lib python3 gen_conv.py
import migraphx
import subprocess
import os
import sys
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(SCRIPT_DIR, "build-debug", "bin", "migraphx-driver")
TRACE_DIR = os.path.join(SCRIPT_DIR, "fuzzer-trace")
LOG_FILE = os.path.join(SCRIPT_DIR, "gen_conv.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

def divisors(n):
    return [i for i in range(1, n + 1) if n % i == 0]

def output_dim(spatial, kernel, pad, stride, dilation):
    return (spatial + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1

def random_conv_params():
    N = random.choice([1, 2, 4])
    C = random.choice([2, 3, 4, 8, 16, 32, 64, 128])
    group = random.choice(divisors(C))
    C_per_group = C // group

    K_choices = [k for k in range(1000) if k % group == 0]
    K = random.choice(K_choices)

    kH = random.choice([1, 2, 3, 5, 7])
    kW = random.choice([1, 2, 3, 5, 7])

    sH = random.choice([1, 2, 3])
    sW = random.choice([1, 2, 3])

    dH = random.choice([1, 2])
    dW = random.choice([1, 2])

    pH = random.randint(0, kH - 1)
    pW = random.randint(0, kW - 1)

    # H >= dH*(kH-1) + 1 - 2*pH  so that output >= 1
    min_H = max(1, dH * (kH - 1) + 1 - 2 * pH)
    min_W = max(1, dW * (kW - 1) + 1 - 2 * pW)

    H = random.randint(min_H, min_H + random.choice([0, 4, 8, 16, 32, 64]))
    W = random.randint(min_W, min_W + random.choice([0, 4, 8, 16, 32, 64]))

    assert output_dim(H, kH, pH, sH, dH) >= 1
    assert output_dim(W, kW, pW, sW, dW) >= 1

    return {
        "stride": [sH, sW],
        "dilation": [dH, dW],
        "group": group,
        "padding": [pH, pW],
        "input_lens": [N, C, H, W],
        "filter_lens": [K, C_per_group, kH, kW],
    }

def make_filename(params):
    s = "x".join(str(v) for v in params["stride"])
    d = "x".join(str(v) for v in params["dilation"])
    g = str(params["group"])
    f = "x".join(str(v) for v in params["filter_lens"])
    i = "x".join(str(v) for v in params["input_lens"])
    return f"conv-{s}-{d}-{g}-{f}-{i}.mxr"

def generate_mxr(params):
    p = migraphx.program()
    mm = p.get_main_module()

    x = mm.add_parameter("x", migraphx.shape(type="float", lens=params["input_lens"]))
    w = mm.add_parameter("w", migraphx.shape(type="float", lens=params["filter_lens"]))

    conv = mm.add_instruction(
        migraphx.op("convolution",
                     **{"stride": params["stride"],
                        "dilation": params["dilation"],
                        "group": params["group"],
                        "padding": params["padding"]}),
        [x, w])
    mm.add_return([conv])

    filename = make_filename(params)
    filepath = os.path.join(TRACE_DIR, filename)
    migraphx.save(p, filepath)
    return filepath, filename

def verify(filepath, filename):
    env = os.environ.copy()
    env["MIGRAPHX_MLIR_USE_SPECIFIC_OPS"] = "convolution"

    cmd = [DRIVER, "verify", "--gpu", "--migraphx", filepath]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    combined = result.stdout + result.stderr
    has_linalg = "Lowering using linalg" in combined
    has_passed = "MIGraphX verification passed successfully." in combined

    label = filename.replace(".mxr", ".mlir")
    if has_linalg and has_passed:
        return True, f"Passed {label}"

    reasons = []
    if not has_linalg:
        reasons.append("Missing: 'Lowering using linalg'")
    if not has_passed:
        reasons.append("Missing: 'MIGraphX verification passed successfully.'")
    return False, f"Failed {label}\n  " + "\n  ".join(reasons)

def run_one_thread(iterations):
    passed = 0
    failed = 0
    for _ in range(iterations):
        params = random_conv_params()
        filepath, filename = generate_mxr(params)
        ok, msg = verify(filepath, filename)
        log.info(msg)
        if ok:
            passed += 1
        else:
            failed += 1
    return passed, failed


def main():
    os.makedirs(TRACE_DIR, exist_ok=True)

    total = 4000
    num_workers = 2 
    per_thread = total // num_workers

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(run_one_thread, per_thread) for _ in range(num_workers)]
        total_passed = 0
        total_failed = 0
        for future in as_completed(futures):
            p, f = future.result()
            total_passed += p
            total_failed += f

    log.info("=" * 40)
    log.info(f"Total: {total}  Passed: {total_passed}  Failed: {total_failed}")
    sys.exit(1 if total_failed else 0)

if __name__ == "__main__":
    main()