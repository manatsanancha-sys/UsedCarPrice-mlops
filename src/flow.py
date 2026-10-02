"""Pipeline แบบ DAG ด้วย Prefect: ข้อมูลดิบ -> ตรวจสอบ -> เทรน -> อนุมัติ (gate) -> export -> เฝ้าระวัง (-> ให้บริการ)

DAG (ลูกศร = ต้องรอ task ก่อนหน้าสำเร็จ):

    download -> split -> validate -> train -> register(gate) -> export -> [serve]
                  |                                  |
                  +-> data_drift                     +-> concept_drift

- แต่ละ task เรียกสคริปต์เดิมใน src/ ด้วย sys.executable (ผลเหมือนรันทีละคำสั่ง)
- task ล้ม (exit != 0) -> Prefect retry ไม่ได้ช่วย -> task ปลายน้ำไม่ถูกรัน, flow = Failed
- drift monitor exit 1 = "พบ drift" (แจ้งเตือน) ไม่ใช่ความผิดพลาด -> คืนผลแทนการทำให้ flow ล้ม
รัน: python -m src.flow  (หรือ python -m src.pipeline)  ·  เพิ่ม --serve เพื่อ build/run Docker ต่อท้าย
"""
import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

RAW = Path("data/raw/car_details_v3.csv")


def run_step(args, ok_codes=(0,)):
    """รันคำสั่ง แสดง output ใน log ของ Prefect; exit code นอก ok_codes -> raise ให้ task ล้ม"""
    logger = get_run_logger()
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in (result.stdout + result.stderr).splitlines():
        if line.strip() and "WARNING" not in line:
            logger.info(line)
    if result.returncode not in ok_codes:
        raise RuntimeError(f"{' '.join(args)} -> exit {result.returncode}")
    return result.returncode


def py(module):
    return [sys.executable, "-m", module]


@task(name="download_data")
def download_data():
    if RAW.exists():
        get_run_logger().info(f"มี {RAW} อยู่แล้ว ข้ามการดาวน์โหลด")
        return
    run_step([sys.executable, "scripts/download_data.py"])


@task(name="split_data")
def split_data():
    run_step(py("src.data_split"))


@task(name="validate_data")
def validate_data():
    run_step(py("src.validation"))


@task(name="train_models")
def train_models():
    run_step(py("src.train"))


@task(name="register_gate")
def register_gate():
    run_step(py("src.register"))  # gate ไม่ผ่าน -> exit 1 -> export/serve ไม่ถูกรัน


@task(name="export_model")
def export_model():
    run_step(py("src.export_model"))


@task(name="data_drift")
def data_drift():
    code = run_step(py("src.monitor_drift"), ok_codes=(0, 1))
    return "DETECTED" if code == 1 else "OK"


@task(name="concept_drift")
def concept_drift():
    code = run_step(py("src.monitor_concept_drift"), ok_codes=(0, 1))
    return "DETECTED" if code == 1 else "OK"


@task(name="serve_docker")
def serve_docker():
    run_step(["docker", "build", "-t", "usedcar-api", "."])
    subprocess.run(["docker", "rm", "-f", "usedcar-api"], capture_output=True)
    run_step(["docker", "run", "-d", "--name", "usedcar-api", "-p", "8000:8000", "usedcar-api"])


@flow(name="used-car-price-pipeline", log_prints=True)
def pipeline(serve: bool = False):
    downloaded = download_data.submit()
    split = split_data.submit(wait_for=[downloaded])
    validated = validate_data.submit(wait_for=[split])
    trained = train_models.submit(wait_for=[validated])
    registered = register_gate.submit(wait_for=[trained])
    exported = export_model.submit(wait_for=[registered])

    # สาขาเฝ้าระวัง รันขนานกับขั้นที่เหลือ
    drift = data_drift.submit(wait_for=[split])
    concept = concept_drift.submit(wait_for=[registered])

    main_steps = {"download_data": downloaded, "split_data": split, "validate_data": validated,
                  "train_models": trained, "register_gate": registered, "export_model": exported}
    if serve:
        main_steps["serve_docker"] = serve_docker.submit(wait_for=[exported])

    for future in [*main_steps.values(), drift, concept]:
        future.wait()
    monitoring = {name: f.result() if f.state.is_completed() else "NOT RUN"
                  for name, f in [("data_drift", drift), ("concept_drift", concept)]}
    print(f"monitoring: data_drift={monitoring['data_drift']}  concept_drift={monitoring['concept_drift']}")

    # ขั้นแรกที่ไม่สำเร็จคือต้นเหตุ ขั้นหลังจากนั้นไม่ถูกรัน (NotReady)
    failed = [name for name, f in main_steps.items() if not f.state.is_completed()]
    if failed:
        skipped = f" (ไม่ถูกรัน: {', '.join(failed[1:])})" if failed[1:] else ""
        raise RuntimeError(f"หยุดที่ขั้น {failed[0]}{skipped}")
    return monitoring


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="build + run Docker API หลัง export")
    args = parser.parse_args()
    try:
        pipeline(serve=args.serve)
    except Exception as e:
        print(f"PIPELINE FAILED: {e}")
        sys.exit(1)
    print("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
