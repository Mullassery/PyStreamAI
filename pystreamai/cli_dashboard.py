"""
PyStreamAI CLI Dashboard - Real-time deployment monitoring

Shows: models, latency, errors, cost tracking, resource usage
"""

import sys
import platform
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DashboardMetrics:
    """Standard metrics container"""
    timestamp: str
    title: str
    metrics: Dict[str, Any]
    alerts: list
    recommendations: list


def get_dashboard_impl(product_name: str):
    """Get platform-specific dashboard implementation"""
    platform_name = platform.system()

    if platform_name == "Darwin":  # macOS
        try:
            from rich.console import Console
            return RichDashboard(product_name)
        except ImportError:
            return SimpleDashboard(product_name)

    elif platform_name == "Linux":
        # Try Textual first, fallback to Rich
        try:
            from textual.app import App
            return TextualDashboard(product_name)
        except ImportError:
            try:
                from rich.console import Console
                return RichDashboard(product_name)
            except ImportError:
                return SimpleDashboard(product_name)

    else:  # Windows or other
        try:
            from rich.console import Console
            return RichDashboard(product_name)
        except ImportError:
            return SimpleDashboard(product_name)


class SimpleDashboard:
    """Fallback plain-text dashboard"""

    def __init__(self, product_name: str):
        self.product_name = product_name

    def render(self, data: DashboardMetrics) -> None:
        print(f"\n{'='*80}")
        print(f"✓ {data.title}")
        print(f"  {data.timestamp}")
        print(f"{'='*80}\n")

        print("KEY METRICS:")
        for key, value in data.metrics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

        if data.alerts:
            print("\n⚠️  ALERTS:")
            for alert in data.alerts:
                print(f"  [{alert.get('level', '').upper()}] {alert.get('message', '')}")

        if data.recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for rec in data.recommendations:
                print(f"  [{rec.get('type', '').upper()}] {rec.get('message', '')}")

        print(f"\n{'='*80}\n")

    def run(self) -> None:
        sample_data = DashboardMetrics(
            timestamp=datetime.now().isoformat(),
            title=f"{self.product_name} Dashboard",
            metrics={"Status": "Active"},
            alerts=[],
            recommendations=[]
        )
        self.render(sample_data)


class RichDashboard:
    """Rich-based dashboard (macOS and Windows primary)"""

    def __init__(self, product_name: str):
        self.product_name = product_name
        try:
            from rich.console import Console
            self.console = Console()
        except ImportError:
            print("Error: Rich library required. Install with: pip install rich")
            sys.exit(1)

    def render(self, data: DashboardMetrics) -> None:
        from rich.table import Table
        from rich.panel import Panel

        self.console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
        self.console.print(f"[bold cyan]✓ {data.title}[/bold cyan]")
        self.console.print(f"[dim cyan]{data.timestamp}[/dim cyan]")
        self.console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")

        # Metrics table
        table = Table(title="[bold]Key Metrics[/bold]")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in data.metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    table.add_row(f"  {key} → {k}", str(v))
            else:
                table.add_row(key, str(value))

        self.console.print(table)

        # Alerts
        if data.alerts:
            self.console.print("\n[bold red]⚠️  ALERTS[/bold red]")
            for alert in data.alerts:
                level = alert.get("level", "info").upper()
                msg = alert.get("message", "")
                self.console.print(f"  [{level}] {msg}")

        # Recommendations
        if data.recommendations:
            self.console.print("\n[bold yellow]💡 RECOMMENDATIONS[/bold yellow]")
            for rec in data.recommendations:
                rec_type = rec.get("type", "").upper()
                msg = rec.get("message", "")
                self.console.print(f"  [{rec_type}] {msg}")

        self.console.print(f"\n[bold cyan]{'='*80}[/bold cyan]\n")

    def run(self) -> None:
        sample_data = DashboardMetrics(
            timestamp=datetime.now().isoformat(),
            title=f"{self.product_name} Dashboard",
            metrics={"Status": "Active ✓", "Uptime": "2h 45m"},
            alerts=[],
            recommendations=[]
        )
        self.render(sample_data)


class TextualDashboard:
    """Textual-based interactive dashboard (Linux)"""

    def __init__(self, product_name: str):
        self.product_name = product_name
        self.has_textual = False
        try:
            from textual.app import App
            self.has_textual = True
        except ImportError:
            pass

    def render(self, data: DashboardMetrics) -> None:
        if not self.has_textual:
            # Fallback to Rich
            dash = RichDashboard(self.product_name)
            dash.render(data)
            return

        # For now, use Rich output as well
        dash = RichDashboard(self.product_name)
        dash.render(data)

    def run(self) -> None:
        if not self.has_textual:
            dash = RichDashboard(self.product_name)
            dash.run()
            return

        sample_data = DashboardMetrics(
            timestamp=datetime.now().isoformat(),
            title=f"{self.product_name} Dashboard",
            metrics={"Status": "Active ✓"},
            alerts=[],
            recommendations=[]
        )
        self.render(sample_data)


class PyStreamAIDashboard:
    """PyStreamAI-specific dashboard implementation"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "./pystreamai.yaml"
        self.dashboard = get_dashboard_impl("PyStreamAI v0.2.0")

    def get_mock_metrics(self) -> DashboardMetrics:
        """Get sample metrics (replace with real metrics in production)"""
        return DashboardMetrics(
            timestamp=datetime.now().isoformat(),
            title="PyStreamAI Deployment Dashboard",
            metrics={
                "Status": "🟢 Healthy",
                "Uptime": "2 days 14h 32m",
                "Models Active": 3,
                "Endpoints": 5,
                "Total Requests": "1,234,567",
                "Latency Metrics": {
                    "Avg": "124ms",
                    "P99": "287ms",
                    "Min": "45ms",
                    "Max": "1.2s",
                },
                "Error Handling": {
                    "Total Errors": "234",
                    "Error Rate": "0.019%",
                },
                "Cost Metrics": {
                    "24h Cost": "$42.17",
                    "Per 1K Requests": "$0.034",
                    "Monthly Est": "$1,265.10",
                },
                "Model Details": {
                    "gpt-4-turbo": "567K req, 145ms avg",
                    "gpt-3.5-turbo": "412K req, 89ms avg",
                    "claude-3": "255K req, 167ms avg",
                },
                "Resource Usage": {
                    "GPU": "67%",
                    "Memory": "8.2/16 GB",
                    "Network (In)": "125 Mbps",
                    "Network (Out)": "234 Mbps",
                },
            },
            alerts=[
                {"level": "info", "message": "All systems operational"},
                {"level": "warning", "message": "GPU temp: 71°C (normal)"},
            ],
            recommendations=[
                {"type": "scaling", "message": "GPU utilization at 67% - consider increasing batch size"},
                {"type": "cost", "message": "gpt-3.5-turbo is more cost-efficient for this workload"},
            ]
        )

    def run_dashboard(self, interactive: bool = True) -> None:
        """Run the dashboard"""
        try:
            metrics = self.get_mock_metrics()

            if interactive:
                self.dashboard.run()
            else:
                self.dashboard.render(metrics)

        except KeyboardInterrupt:
            print("\n\nDashboard stopped.")
            sys.exit(0)
        except Exception as e:
            print(f"Error running dashboard: {e}", file=sys.stderr)
            sys.exit(1)

    def show_alerts(self) -> None:
        """Show only alerts"""
        metrics = self.get_mock_metrics()
        print("\n[ALERTS]")
        if metrics.alerts:
            for alert in metrics.alerts:
                print(f"  [{alert['level'].upper()}] {alert['message']}")
        else:
            print("  ✓ No alerts")

    def show_recommendations(self) -> None:
        """Show only recommendations"""
        metrics = self.get_mock_metrics()
        print("\n[RECOMMENDATIONS]")
        if metrics.recommendations:
            for rec in metrics.recommendations:
                print(f"  [{rec['type'].upper()}] {rec['message']}")
        else:
            print("  ✓ No recommendations")

    def export_json(self, output_file: str) -> None:
        """Export metrics as JSON"""
        import json
        metrics = self.get_mock_metrics()
        data = {
            "timestamp": metrics.timestamp,
            "title": metrics.title,
            "metrics": metrics.metrics,
            "alerts": metrics.alerts,
            "recommendations": metrics.recommendations,
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Metrics exported to {output_file}")
