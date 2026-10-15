import pytest
import os
import pandas as pd
from marketing_incrementality_mmm.reports.audit_generator import AuditReportGenerator

@pytest.fixture(scope="module")
def mock_audit_data():
    diagnostics = {"r2_score": 0.9053, "mape_pct": 3.57, "rmse": 1914.87}
    df_attr = pd.DataFrame([
        {"Channel": "Google_PMax_NonBrand", "Total Spend ($)": 1445216.24, "Incremental Revenue ($)": 1732912.14, "True iROAS": 1.20, "In-Platform Claimed ROAS": 3.10, "Over-Attribution Gap": "+158.3%"},
        {"Channel": "YouTube_Awareness", "Total Spend ($)": 816360.41, "Incremental Revenue ($)": 1676677.76, "True iROAS": 2.05, "In-Platform Claimed ROAS": 1.45, "Over-Attribution Gap": "-29.3%"}
    ])
    df_realloc = pd.DataFrame([
        {"Channel": "Google_PMax_NonBrand", "Current Spend ($/day)": 1985.19, "Optimal Spend ($/day)": 739.80, "Recommended Shift ($)": -1245.39, "Current mROAS": 0.30, "Optimal mROAS": 0.73, "Incremental Lift ($/day)": -565.66},
        {"Channel": "YouTube_Awareness", "Current Spend ($/day)": 1121.37, "Optimal Spend ($/day)": 1644.80, "Recommended Shift ($)": 523.43, "Current mROAS": 0.96, "Optimal mROAS": 0.73, "Incremental Lift ($/day)": 438.68}
    ])
    return {
        "company_name": "Test Brand DTC",
        "diagnostics": diagnostics,
        "df_attribution": df_attr,
        "df_reallocation": df_realloc,
        "annual_uplift": 173077.24,
        "daily_uplift": 473.86
    }

def test_markdown_generation(mock_audit_data):
    generator = AuditReportGenerator(**mock_audit_data)
    md = generator.generate_markdown()
    assert "# 📑 Marketing Incrementality & Media Mix Audit" in md
    assert "Test Brand DTC" in md
    assert "+$173,077.24 per year" in md
    assert "Google_PMax_NonBrand" in md

def test_html_generation(mock_audit_data):
    generator = AuditReportGenerator(**mock_audit_data)
    html = generator.generate_html()
    assert "<!DOCTYPE html>" in html
    assert "Test Brand DTC" in html
    assert "+$473.86" in html
    assert "0.73x" in html

def test_report_export(mock_audit_data, tmp_path):
    generator = AuditReportGenerator(**mock_audit_data)
    paths = generator.export(output_dir=str(tmp_path))
    assert os.path.exists(paths["html_path"])
    assert os.path.exists(paths["md_path"])
    assert os.path.getsize(paths["html_path"]) > 1000
    assert os.path.getsize(paths["md_path"]) > 500
