from __future__ import annotations

from pathlib import Path

from sas.gui import generator_inputs as generator_inputs_module


def test_build_city_demand_preview_reads_od_and_node_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    city_root = tmp_path / "data" / "cities" / "sample" / "bundle"
    city_root.mkdir(parents=True)
    od_path = city_root / "sample_od.csv"
    node_path = city_root / "sample_node.csv"
    od_path.write_text(
        "O_ID,D_ID,OD_Number\n10000001,10000002,12\n10000002,10000002,5\n",
        encoding="utf-8",
    )
    node_path.write_text(
        "Node_ID,Lon,Lat\n10000001,22.95,40.63\n10000002,22.97,40.64\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(generator_inputs_module, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        generator_inputs_module,
        "_CITIES_ROOT",
        (tmp_path / "data" / "cities").resolve(),
    )
    monkeypatch.setattr(generator_inputs_module.generate_city_module, "PROJECT_ROOT", tmp_path.resolve())

    preview = generator_inputs_module.build_city_demand_preview("sample")

    assert preview["supported"] is True
    assert preview["summary"]["od_row_count"] == 2
    assert preview["summary"]["intrazonal_raw"] == 5
    assert preview["summary"]["mapped_top_flow_count"] == 1
    assert preview["sample_rows"][0]["origin"] == "10000001"
    assert preview["top_flows"][0]["origin_coords"] == [40.63, 22.95]


def test_build_city_demand_preview_reports_missing_files(
    tmp_path: Path, monkeypatch
) -> None:
    city_root = tmp_path / "data" / "cities" / "sample"
    city_root.mkdir(parents=True)

    monkeypatch.setattr(generator_inputs_module, "PROJECT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        generator_inputs_module,
        "_CITIES_ROOT",
        (tmp_path / "data" / "cities").resolve(),
    )
    monkeypatch.setattr(generator_inputs_module.generate_city_module, "PROJECT_ROOT", tmp_path.resolve())

    preview = generator_inputs_module.build_city_demand_preview("sample")

    assert preview["supported"] is False
    assert preview["issues"]
