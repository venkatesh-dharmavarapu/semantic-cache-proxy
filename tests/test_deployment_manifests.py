import os
import yaml


def test_helm_chart_yaml_structure():
    chart_path = os.path.join("deploy", "helm", "semantic-cache-proxy", "Chart.yaml")
    assert os.path.exists(chart_path), "Chart.yaml must exist"

    with open(chart_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["name"] == "semantic-cache-proxy"
    assert data["apiVersion"] == "v2"
    assert "version" in data


def test_helm_values_yaml_keys():
    values_path = os.path.join("deploy", "helm", "semantic-cache-proxy", "values.yaml")
    assert os.path.exists(values_path), "values.yaml must exist"

    with open(values_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "replicaCount" in data
    assert "resources" in data
    assert "autoscaling" in data
    assert "podDisruptionBudget" in data
    assert data["env"]["REDIS_URL"] is not None
    assert data["probes"]["liveness"]["path"] == "/health/live"
    assert data["probes"]["readiness"]["path"] == "/health/ready"


def test_github_actions_workflow_yaml():
    workflow_path = os.path.join(".github", "workflows", "ci-cd.yaml")
    assert os.path.exists(workflow_path), "ci-cd.yaml must exist"

    with open(workflow_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "jobs" in data
    assert "test" in data["jobs"]
    assert "helm-lint" in data["jobs"]