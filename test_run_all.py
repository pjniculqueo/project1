from main import run_all


def test_run_all_runs(tmp_path, monkeypatch):
    # Ensure plots go to temporary directory
    monkeypatch.chdir(tmp_path)
    var99, avg = run_all(num_scenarios=10)
    assert isinstance(var99, float)
    assert isinstance(avg, float)
    assert any(p.suffix == '.png' for p in tmp_path.iterdir())
