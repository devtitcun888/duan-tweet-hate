"""Capture 4 real screenshots of the running Streamlit app (app/app.py) for P10 and Chapter 7 of the report."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "reports" / "figures"


def _wait_for_server(url: str, timeout: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def capture_screenshots() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    py_exe = str(venv_py) if venv_py.exists() else sys.executable

    proc = subprocess.Popen(
        [
            py_exe,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "app" / "app.py"),
            "--server.port",
            "8511",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        ok = _wait_for_server("http://localhost:8511", timeout=30.0)
        if not ok:
            raise RuntimeError("Streamlit server did not start on port 8511 within 30s")

        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1480,980")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--hide-scrollbars")

        driver = webdriver.Chrome(options=opts)
        try:
            driver.get("http://localhost:8511")
            time.sleep(5.0)

            # Screenshot 1: Tab 1 Default Normal Tweet
            driver.save_screenshot(str(FIG_DIR / "P10_app_tab1_normal.png"))
            print("Saved P10_app_tab1_normal.png")

            # Click sample button 4 (or 3) for Hate Tweet on Tab 1
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for b in buttons:
                if "3. Phản cảm" in b.text or "4. Phản cảm" in b.text:
                    b.click()
                    time.sleep(2.5)
                    break
            driver.save_screenshot(str(FIG_DIR / "P10_app_tab1_hate.png"))
            print("Saved P10_app_tab1_hate.png")

            # Switch to Tab 2 (Batch CSV)
            tab2_els = driver.find_elements(By.XPATH, "//*[contains(text(), 'Tab 2')]")
            if tab2_els:
                driver.execute_script("arguments[0].click();", tab2_els[0])
                time.sleep(2.5)
                buttons_t2 = driver.find_elements(By.XPATH, "//*[contains(text(), '100 mẫu Thực tế')]")
                if buttons_t2:
                    driver.execute_script("arguments[0].click();", buttons_t2[0])
                    time.sleep(3.5)
                driver.save_screenshot(str(FIG_DIR / "P10_app_tab2_batch.png"))
                print("Saved P10_app_tab2_batch.png")

            # Switch to Tab 3 (Stats & Model Card)
            tab3_els = driver.find_elements(By.XPATH, "//*[contains(text(), 'Tab 3')]")
            if tab3_els:
                driver.execute_script("arguments[0].click();", tab3_els[0])
                time.sleep(3.0)
                driver.save_screenshot(str(FIG_DIR / "P10_app_tab3_stats.png"))
                print("Saved P10_app_tab3_stats.png")

        finally:
            driver.quit()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    capture_screenshots()
