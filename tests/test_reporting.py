import pytest
import os
from src.reporting import Reporting

@pytest.fixture
def reporting():
    return Reporting()

def test_add_metric(reporting):
    reporting.add_metric('accuracy', 0.85)
    assert 'accuracy' in reporting.report_data
    assert reporting.report_data['accuracy'] == [0.85]

def test_generate_text_report(reporting):
    reporting.add_metric('accuracy', 0.85)
    reporting.add_metric('loss', 0.15)
    report = reporting.generate_text_report()
    assert 'Model Deployment Report' in report
    assert 'accuracy' in report
    assert 'loss' in report

def test_generate_visual_report(reporting):
    reporting.add_metric('accuracy', 0.85)
    reporting.add_metric('loss', 0.15)
    reporting.generate_visual_report('test_report.png')
    assert os.path.exists('test_report.png')
    os.remove('test_report.png')

def test_save_report_to_file(reporting):
    reporting.add_metric('accuracy', 0.85)
    reporting.save_report_to_file('test_report.txt')
    assert os.path.exists('test_report.txt')
    with open('test_report.txt', 'r') as f:
        content = f.read()
    assert 'Model Deployment Report' in content
    os.remove('test_report.txt')

def test_generate_csv_report(reporting):
    reporting.add_metric('accuracy', 0.85)
    reporting.add_metric('loss', 0.15)
    reporting.generate_csv_report('test_report.csv')
    assert os.path.exists('test_report.csv')
    os.remove('test_report.csv')
