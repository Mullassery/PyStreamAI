"""
PyStreamAI CLI - Command-line interface for PyStreamAI
"""

import sys
import argparse
from datetime import datetime
from typing import Optional
from pathlib import Path


def dashboard_command(args):
    """Handle dashboard command"""
    try:
        from pystreamai.cli_dashboard import PyStreamAIDashboard

        dashboard = PyStreamAIDashboard(config_path=args.config)

        if args.export:
            dashboard.export_json(args.export)
        elif args.alerts:
            dashboard.show_alerts()
        elif args.recommendations:
            dashboard.show_recommendations()
        else:
            dashboard.run_dashboard(interactive=not args.static)

    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="PyStreamAI - Production ML Deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pystreamai dashboard              # Interactive dashboard
  pystreamai dashboard --static     # Static view
  pystreamai dashboard --alerts     # Show alerts only
  pystreamai dashboard --export metrics.json  # Export metrics

For more help: https://github.com/mullassery/pystreamai#quickstart
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Dashboard subcommand
    dashboard_parser = subparsers.add_parser(
        'dashboard',
        help='View real-time deployment dashboard'
    )
    dashboard_parser.add_argument(
        '--static',
        action='store_true',
        help='Show static snapshot (non-interactive)'
    )
    dashboard_parser.add_argument(
        '--alerts',
        action='store_true',
        help='Show alerts only'
    )
    dashboard_parser.add_argument(
        '--recommendations',
        action='store_true',
        help='Show recommendations only'
    )
    dashboard_parser.add_argument(
        '--export',
        metavar='FILE',
        help='Export metrics to JSON file'
    )
    dashboard_parser.add_argument(
        '--config',
        metavar='PATH',
        help='Path to pystreamai config file'
    )
    dashboard_parser.set_defaults(func=dashboard_command)

    # Version
    parser.add_argument(
        '--version',
        action='version',
        version='PyStreamAI 0.2.0'
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, 'func'):
        args.func(args)


if __name__ == '__main__':
    main()
