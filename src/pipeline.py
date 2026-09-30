"""รัน pipeline ทั้งหมดตั้งแต่ข้อมูลดิบจนถึงโมเดล champion เรียงลำดับ"""
import subprocess
import sys

STEPS = [
    [sys.executable, "-m", "src.data_split"],
    [sys.executable, "-m", "src.validation"],
    [sys.executable, "-m", "src.train"],
    [sys.executable, "-m", "src.register"],
    [sys.executable, "-m", "src.export_model"],
]


def main():
    for step in STEPS:
        print(f"\n=== RUN: {' '.join(step)} ===")
        result = subprocess.run(step)
        if result.returncode != 0:
            print(f"FAILED at: {' '.join(step)}")
            sys.exit(1)
    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
