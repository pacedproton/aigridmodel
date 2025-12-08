#!/usr/bin/env python3
"""
UI Integration Test for Classical Model Workflow
Tests the complete user journey: clicking buttons, seeing results
"""

import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import requests

class TestClassicalModelUI:
    """Test classical model UI interactions"""

    @pytest.fixture(scope="class")
    def driver(self):
        """Set up Chrome driver for UI testing"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run headless for CI
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        yield driver
        driver.quit()

    @pytest.fixture(scope="class")
    def setup_services(self):
        """Ensure backend and frontend are running"""
        # Check backend
        try:
            response = requests.get("http://localhost:5001/api/health", timeout=5)
            assert response.status_code == 200
        except:
            pytest.skip("Backend not running")

        # Check frontend
        try:
            response = requests.get("http://localhost:60000", timeout=5)
            assert response.status_code == 200
        except:
            pytest.skip("Frontend not running")

    def test_classical_model_workflow(self, driver, setup_services):
        """Test complete classical model workflow"""
        driver.get("http://localhost:60000")

        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "h4"))
        )

        # Select forecasting use case (should be default or click tab)
        forecasting_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load/Demand Forecasting')]"))
        )
        forecasting_tab.click()

        # Click "Classical Model" button
        classical_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Classical Model')]"))
        )
        classical_button.click()

        # Wait for results to appear
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Classical Model Results')]"))
        )

        # Verify results are displayed
        results_section = driver.find_element(By.XPATH, "//*[contains(text(), 'Classical Model Results')]")
        assert results_section.is_displayed()

        # Check for metrics (MSE, MAE, etc.)
        metrics_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'MSE:') or contains(text(), 'MAE:')]")
        assert len(metrics_elements) > 0, "No metrics found in results"

        print("✅ Classical model UI workflow test passed")

    def test_advanced_model_workflow(self, driver, setup_services):
        """Test advanced model workflow"""
        driver.get("http://localhost:60000")

        # Select forecasting use case
        forecasting_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load/Demand Forecasting')]"))
        )
        forecasting_tab.click()

        # Click "Advanced Math" button
        advanced_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Advanced Math')]"))
        )
        advanced_button.click()

        # Wait for results
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Advanced Math Results')]"))
        )

        # Verify Bayesian forecasting results
        results_section = driver.find_element(By.XPATH, "//*[contains(text(), 'Advanced Math Results')]")
        assert results_section.is_displayed()

        print("✅ Advanced model UI workflow test passed")

    def test_comparison_workflow(self, driver, setup_services):
        """Test model comparison workflow"""
        driver.get("http://localhost:60000")

        # Select use case
        forecasting_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load/Demand Forecasting')]"))
        )
        forecasting_tab.click()

        # Click "Show Comparison" button
        comparison_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Show Comparison')]"))
        )
        comparison_button.click()

        # Click "Run All Models" button
        run_all_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Run All Models')]"))
        )
        run_all_button.click()

        # Wait for comparison results
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Performance Overview')]"))
        )

        # Verify comparison dashboard appears
        performance_section = driver.find_element(By.XPATH, "//*[contains(text(), 'Performance Overview')]")
        assert performance_section.is_displayed()

        print("✅ Model comparison UI workflow test passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
