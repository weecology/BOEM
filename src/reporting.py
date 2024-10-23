import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class Reporting:
    def __init__(self):
        self.report_data = {}

    def add_metric(self, metric_name, value):
        """Add a metric to the report data."""
        if metric_name not in self.report_data:
            self.report_data[metric_name] = []
        self.report_data[metric_name].append(value)

    def generate_text_report(self):
        """Generate a text-based report."""
        report = "Model Deployment Report\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        for metric, values in self.report_data.items():
            report += f"{metric}:\n"
            report += f"  Latest: {values[-1]}\n"
            report += f"  Average: {sum(values) / len(values):.2f}\n"
            report += f"  Min: {min(values)}\n"
            report += f"  Max: {max(values)}\n\n"

        return report

    def generate_visual_report(self, output_file='report.png'):
        """Generate a visual report with plots."""
        num_metrics = len(self.report_data)
        fig, axs = plt.subplots(num_metrics, 1, figsize=(10, 5*num_metrics))
        
        for i, (metric, values) in enumerate(self.report_data.items()):
            ax = axs[i] if num_metrics > 1 else axs
            ax.plot(values)
            ax.set_title(metric)
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Value')

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def save_report_to_file(self, filename='report.txt'):
        """Save the text report to a file."""
        with open(filename, 'w') as f:
            f.write(self.generate_text_report())

    def generate_csv_report(self, filename='report.csv'):
        """Generate a CSV report."""
        df = pd.DataFrame(self.report_data)
        df.to_csv(filename, index=False)

    def generate_report(self, pipeline_evaluation, model_deployment):
        # Add metrics to the report
        self.add_metric('accuracy', pipeline_evaluation.accuracy)
        self.add_metric('loss', pipeline_evaluation.loss)

        # Generate reports
        self.generate_text_report()
        self.generate_visual_report()
        self.save_report_to_file('deployment_report.txt')
        self.generate_csv_report('deployment_report.csv')
