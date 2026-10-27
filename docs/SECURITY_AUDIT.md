# Security Vulnerability Audit - pystreamai

**Audit Date:** 2026-10-27
**Repository:** /Users/georgimullassery/Desktop/Projects/pystreamai
**Total Findings:** 5

## Summary
- **Critical:** 5

## Findings

### 1. [Critical] Command Injection
- **File:** benchmarks/bench_oss_comparison.py:39
- **Description:** eval()
- **Evidence (Line):** `model = model.to(device).eval()`
- **Exploit Scenario:** Attacker could execute arbitrary system commands
- **Fix Recommendation:** Avoid shell execution, use subprocess with shell=False and list args
- **Confidence:** High

### 2. [Critical] Command Injection
- **File:** benchmarks/models.py:27
- **Description:** eval()
- **Evidence (Line):** `model.eval()`
- **Exploit Scenario:** Attacker could execute arbitrary system commands
- **Fix Recommendation:** Avoid shell execution, use subprocess with shell=False and list args
- **Confidence:** High

### 3. [Critical] Command Injection
- **File:** benchmarks/models.py:52
- **Description:** eval()
- **Evidence (Line):** `model.eval()`
- **Exploit Scenario:** Attacker could execute arbitrary system commands
- **Fix Recommendation:** Avoid shell execution, use subprocess with shell=False and list args
- **Confidence:** High

### 4. [Critical] Command Injection
- **File:** pystreamai/onnx_runtime.py:190
- **Description:** eval()
- **Evidence (Line):** `pytorch_model.eval()`
- **Exploit Scenario:** Attacker could execute arbitrary system commands
- **Fix Recommendation:** Avoid shell execution, use subprocess with shell=False and list args
- **Confidence:** High

### 5. [Critical] Insecure Deserialization
- **File:** tests/test_platform.py:111
- **Description:** pickle load/dumps
- **Evidence (Line):** `artifact = pickle.load(f)`
- **Exploit Scenario:** Attacker could achieve remote code execution
- **Fix Recommendation:** Use safe serialization formats (JSON), avoid pickle/yaml.load with untrusted data
- **Confidence:** High
