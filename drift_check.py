import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

df = pd.read_csv("data/salary_data_analyst.csv")

split = len(df) // 2
reference = df.iloc[:split]
current = df.iloc[split:]

report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=reference, current_data=current)
result.save_html("charts/drift_report.html")
print("Drift report saved to charts/drift_report.html")