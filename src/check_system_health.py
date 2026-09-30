
"""ตรวจสถานะระบบ (System Health): เรียก /health และ /metrics ของ API
แล้วประเมินตาม SLO ที่ตั้งไว้
"""
import sys
import requests

API_URL = "http://127.0.0.1:8000"

# เกณฑ์ SLO
MAX_P95_LATENCY_MS = 250
MAX_ERROR_RATE = 0.01


def check_health():
    r = requests.get(f"{API_URL}/health", timeout=5)
    r.raise_for_status()
    data = r.json()
    ok = data.get("status") == "ok"
    print(f"[health] status={data.get('status')} model_version={data.get('model_version')} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_metrics():
    r = requests.get(f"{API_URL}/metrics", timeout=5)
    r.raise_for_status()
    data = r.json()
    total = data.get("total_requests", 0)
    errors = data.get("total_errors", 0)
    p95 = data.get("p95_latency_ms")
    error_rate = errors / total if total else 0

    print(f"[metrics] total_requests={total} errors={errors} "
          f"error_rate={error_rate:.1%} p95_latency={p95}")

    ok = True
    if p95 is not None and p95 > MAX_P95_LATENCY_MS:
        print(f"  ALERT: p95 latency {p95:.0f}ms เกิน SLO ({MAX_P95_LATENCY_MS}ms)")
        ok = False
    if error_rate > MAX_ERROR_RATE:
        print(f"  ALERT: error rate {error_rate:.1%} เกิน SLO ({MAX_ERROR_RATE:.0%})")
        ok = False
    return ok


def main():
    health_ok = check_health()
    metrics_ok = check_metrics()
    if health_ok and metrics_ok:
        print("\nSYSTEM STATUS: HEALTHY")
        sys.exit(0)
    else:
        print("\nSYSTEM STATUS: UNHEALTHY - ต้องแจ้งเตือนทีม")
        sys.exit(1)


if __name__ == "__main__":
    main()
