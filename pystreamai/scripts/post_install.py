"""
Post-installation message for PyStreamAI
Displays when user runs: pip install pystreamai
"""


def post_install():
    """Display post-install message"""
    message = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ PyStreamAI v0.2.0 installed successfully!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 WHAT IS PyStreamAI?
   Production ML deployment platform with automatic versioning, intelligent
   rollbacks, and cost optimization for LLM inference at scale.

🚀 GET STARTED IN 2 MINUTES:

   Step 1 — Deploy your first model:
   $ pystreamai deploy --model gpt-4

   Step 2 — Start the inference server:
   $ pystreamai serve --port 8000

   Step 3 — View real-time dashboard:
   $ pystreamai dashboard

📚 KEY FEATURES YOU CAN DO:
   • Deploy LLM models to production with auto-versioning and rollback
   • Route requests across multiple providers (OpenAI, Anthropic, etc.)
   • Track inference latency, error rates, and costs in real-time
   • Auto-scale deployments based on load and cost constraints
   • Multi-cloud deployment (AWS, GCP, Azure, on-prem)

📊 VIEW DASHBOARD:
   $ pystreamai dashboard              # Interactive dashboard
   $ pystreamai dashboard --static     # Static snapshot
   $ pystreamai dashboard --alerts     # Show alerts only

📖 LEARN MORE:
   Quick Start:  https://github.com/mullassery/pystreamai#quickstart
   Examples:     https://github.com/mullassery/pystreamai/tree/main/examples
   Issues:       https://github.com/mullassery/pystreamai/issues

❓ GET HELP ANYTIME:
   $ pystreamai --help
   $ pystreamai --version
   $ pystreamai deploy --help         # Help for specific command

⏱️  NEXT STEP: Run `pystreamai deploy --model gpt-4` to deploy your first model!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(message)


if __name__ == "__main__":
    post_install()
