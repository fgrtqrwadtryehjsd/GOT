"""Wait for qasper to finish, then launch 2wikimqa run.

Polls the qasper log file for completion, then starts run_parallel on 2wikimqa
n=200 with cot_sc + gers_cv2_fullctx.

Usage: python experiments/_wait_and_run_2wikimqa.py
"""
import time
import subprocess
import sys
from pathlib import Path

QASPER_LOG = Path("experiments/results/longbench_qasper_8b/_run_log.txt")
DONE_MARKER = "gers_cv2_fullctx] 完成"
OUT_DIR = Path("experiments/results/longbench_2wikimqa_8b/")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_qasper():
    print("[wait] polling qasper log for completion marker...")
    while True:
        if QASPER_LOG.exists():
            content = QASPER_LOG.read_text(encoding="utf-8", errors="replace")
            if DONE_MARKER in content:
                print("[wait] qasper CV2 finished, launching 2wikimqa in 10s...")
                time.sleep(10)
                return
        time.sleep(30)


def launch_2wikimqa():
    cmd = [
        sys.executable, "-u", "experiments/run_parallel.py",
        "--dataset", "longbench_2wikimqa",
        "--methods", "cot_sc,gers_cv2_fullctx",
        "--num_samples", "200",
        "--workers", "4",
        "--context_field", "context_full",
        "--timeout", "300",
        "--output_dir", str(OUT_DIR),
    ]
    log_out = OUT_DIR / "_run_log.txt"
    log_err = OUT_DIR / "_err_log.txt"
    print(f"[launch] cmd = {' '.join(cmd)}")
    print(f"[launch] log -> {log_out}")
    with open(log_out, "w", encoding="utf-8") as f_out, \
         open(log_err, "w", encoding="utf-8") as f_err:
        proc = subprocess.Popen(cmd, stdout=f_out, stderr=f_err,
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                                if sys.platform == "win32" else 0)
        print(f"[launch] pid = {proc.pid}")
        Path("experiments/results/_active_pids.json").write_text(
            f'{{"2wikimqa_pid": {proc.pid}}}', encoding="utf-8")
        # Do not wait; let it run in background


def main():
    wait_for_qasper()
    launch_2wikimqa()
    print("[done] 2wikimqa launched in background. This wrapper exits now.")


if __name__ == "__main__":
    main()
