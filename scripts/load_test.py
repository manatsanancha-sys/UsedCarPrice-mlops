import concurrent.futures as cf
import time

import requests

URL = "http://127.0.0.1:8000/predict"
BODY = {
    "name": "Maruti Swift Dzire VDI", "year": 2014, "km_driven": 145500,
    "fuel": "Diesel", "seller_type": "Individual", "transmission": "Manual",
    "owner": "First Owner", "mileage": "23.4 kmpl", "engine": "1248 CC",
    "max_power": "74 bhp", "torque": "190Nm@ 2000rpm", "seats": 5,
}


def one(_):
    t = time.perf_counter()
    r = requests.post(URL, json=BODY, timeout=10)
    return (time.perf_counter() - t) * 1000, r.status_code


def run(n, workers):
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, range(n)))
    dur = time.perf_counter() - t0
    lat = sorted(x[0] for x in res)
    ok = sum(1 for x in res if x[1] == 200)

    def pct(q):
        return lat[min(len(lat) - 1, int(q * len(lat)))]

    print(f"workers={workers:2d} n={n} ok={ok} "
          f"p50={pct(.5):.1f}ms p95={pct(.95):.1f}ms p99={pct(.99):.1f}ms "
          f"throughput={n / dur:.1f} req/s")


if __name__ == "__main__":
    for _ in range(20):  # warm-up
        one(0)
    run(300, 1)
    run(300, 10)
