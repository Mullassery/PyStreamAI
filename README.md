# PyStreamAI

Deploy and manage inference workloads across cloud providers. Real-time cost tracking, multi-model orchestration, endpoint management. 40-50x inference speedup with Rust+Python.

**Latest Version:** 0.2.1

## Features

- ✅ Multi-cloud deployment (AWS/GCP/Azure/on-prem)
- ✅ 40-50x inference speedup (Rust+Python)
- ✅ Real-time cost tracking (24h, per 1K, monthly)
- ✅ Production-ready CLI dashboards
- ✅ Keyboard shortcuts for quick access
- ✅ OpenTelemetry support (6 backends)
- ✅ Cross-platform (macOS/Linux/Windows)

## Installation

```bash
pip install pystreamai
```

## Quick Start

```bash
# Setup keyboard shortcuts (one-time)
bash scripts/setup_shortcuts.sh

# View dashboard
dash-pystreamai              # Static snapshot
dash-pystreamai-live         # Live monitoring
dash-pystreamai-export       # Export metrics

# Start deployment
pystreamai deploy --config models.yaml
pystreamai serve --port 8000
```

## Dashboard

Access real-time metrics:
- `dash-pystreamai` - View deployment metrics snapshot
- `dash-pystreamai-live` - Watch inference latency, costs, errors in real-time
- `dash-pystreamai-export` - Export to JSON for integration

**Metrics tracked:** Status, Uptime, Models Active, Endpoints, Requests, Latency (avg/p99/min/max), Errors, Cost

See `DASHBOARD_SHORTCUTS.md` for complete documentation.

## OpenTelemetry

Export metrics to 6 monitoring backends (2-5 min setup):

```bash
# Prometheus (OSS)
export OTEL_EXPORTER_OTLP_PROTOCOL=prometheus
dash-pystreamai-live
curl http://localhost:8000/metrics

# Datadog (Enterprise)
export DD_API_KEY="your-key"
export OTEL_EXPORTER_OTLP_PROTOCOL=datadog
dash-pystreamai-live

# Honeycomb / New Relic / Jaeger / X-Ray also supported
```

See `OTEL_SETUP_GUIDE.md` for all 6 backends.

## Production Deployment

Ready for Kubernetes and Docker:

```bash
# Kubernetes
kubectl apply -f PRODUCTION_DEPLOYMENT.md

# Docker Compose
docker-compose up -d
```

Complete K8s manifests, Docker Compose stack, health checks included.

See `PRODUCTION_DEPLOYMENT.md` for deployment patterns.

## Documentation

- `DASHBOARD_SHORTCUTS.md` - Keyboard shortcuts reference
- `OTEL_SETUP_GUIDE.md` - OpenTelemetry backend setup (6 options)
- `PRODUCTION_DEPLOYMENT.md` - K8s/Docker deployment patterns

## Repository

- GitHub: https://github.com/Mullassery/PyStreamAI
- PyPI: https://pypi.org/project/pystreamai
- Issues: https://github.com/Mullassery/PyStreamAI/issues

## License

MIT
