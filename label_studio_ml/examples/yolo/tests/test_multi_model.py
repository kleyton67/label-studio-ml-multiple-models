"""
Tests for multi-model support.

A control tag can reference several comma-separated models via `model_path`
(e.g. model_path="plate_id_1.pt,container_v12.pt"). Their predictions are merged
into a single result, and only one model is kept on the GPU at a time.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest

from control_models.base import ControlModel, parse_model_paths
from control_models.rectangle_labels import RectangleLabelsModel
from .test_common import client


def make_control_model(model_paths):
    control = type("Control", (), {})()
    control.name = "label"
    control.to_name = ["image"]
    control.objects = [type("Object", (), {"value_name": "$image"})()]
    control.attr = {}
    control.tag = "RectangleLabels"
    mlbackend = type("MLBackend", (), {})()
    mlbackend.project_id = "42"
    mlbackend.label_interface = None
    return RectangleLabelsModel.model_construct(
        control=control,
        from_name="label",
        to_name="image",
        value="$image",
        model=None,
        model_paths=list(model_paths),
        label_maps={path: {"car": "Car"} for path in model_paths},
        label_map={"car": "Car"},
        label_studio_ml_backend=mlbackend,
        project_id="42",
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("plate_id_1.pt,container_v12.pt", ["plate_id_1.pt", "container_v12.pt"]),
        ("plate_id_1.pt, container_v12.pt", ["plate_id_1.pt", "container_v12.pt"]),
        ("plate_id_1.pt", ["plate_id_1.pt"]),
        ("", []),
        (None, []),
        ("a.pt,,b.pt,", ["a.pt", "b.pt"]),
    ],
)
def test_parse_model_paths(raw, expected):
    assert parse_model_paths(raw) == expected


def test_multi_model_merges_regions_and_offloads_one_at_a_time():
    instance = make_control_model(["m1.pt", "m2.pt"])
    events = []

    def fake_get_cached(path):
        model = type("Model", (), {"_path": path, "names": {0: "car"}})()
        events.append(("load", path))
        return model

    def fake_predict_regions(path):
        events.append(("predict", instance.model._path, path))
        return [{"from_name": "label", "score": 0.9}]

    def fake_offload(model):
        events.append(("offload", model._path))

    with patch.object(
        RectangleLabelsModel, "get_cached_model", side_effect=fake_get_cached
    ), patch.object(
        RectangleLabelsModel, "offload_model", side_effect=fake_offload
    ), patch.object(RectangleLabelsModel, "predict_regions", side_effect=fake_predict_regions):
        results = instance.predict_regions_for_tasks(["p1", "p2"])

    assert events == [
        ("load", "m1.pt"),
        ("predict", "m1.pt", "p1"),
        ("predict", "m1.pt", "p2"),
        ("offload", "m1.pt"),
        ("load", "m2.pt"),
        ("predict", "m2.pt", "p1"),
        ("predict", "m2.pt", "p2"),
        ("offload", "m2.pt"),
    ]
    assert [len(r) for r in results] == [2, 2]


def test_single_model_does_not_offload():
    instance = make_control_model(["m1.pt"])
    offload_calls = []

    with patch.object(
        RectangleLabelsModel, "offload_model", side_effect=lambda m: offload_calls.append(m)
    ), patch.object(
        RectangleLabelsModel, "predict_regions", side_effect=lambda path: [{"score": 0.9}]
    ):
        results = instance.predict_regions_for_tasks(["p1", "p2"])

    assert [len(r) for r in results] == [1, 1]
    assert offload_calls == []


def test_multi_model_predict_merges_distinct_detections(client):
    from ultralytics import YOLO

    class FakeBoxes:
        def __init__(self, conf, cls, xywhn):
            self.conf = conf
            self.cls = cls
            self.xywhn = xywhn

        @property
        def shape(self):
            return (len(self.conf),)

    class FakeResult:
        def __init__(self, boxes):
            self.boxes = boxes
            self.obb = None

        def plot(self):
            return None

    model_a = YOLO("models/yolov8n.pt")
    model_b = YOLO("models/yolov8n.pt")
    model_a.predict = lambda path: [
        FakeResult(FakeBoxes([0.9], [2], np.array([[0.5, 0.5, 0.2, 0.2]])))
    ]
    model_b.predict = lambda path: [
        FakeResult(FakeBoxes([0.8], [5], np.array([[0.1, 0.1, 0.3, 0.3]])))
    ]

    label_config = """
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image" model_score_threshold="0.25"
                      model_path="model_a.pt,model_b.pt">
        <Label value="Car" predicted_values="car"/>
        <Label value="Bus" predicted_values="bus"/>
      </RectangleLabels>
    </View>
    """
    task = {
        "data": {
            "image": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg"
        }
    }

    with patch.object(
        ControlModel,
        "get_cached_model",
        classmethod(lambda cls, path: model_a if path == "model_a.pt" else model_b),
    ):
        response = client.post(
            "/setup",
            data=json.dumps({"schema": label_config, "project": "42"}),
            content_type="application/json",
        )
        assert response.status_code == 200, response.content

        response = client.post(
            "/predict",
            data=json.dumps({"tasks": [task], "label_config": label_config}),
            content_type="application/json",
        )
        assert response.status_code == 200, response.content

    result = response.json["results"][0]["result"]
    labels = sorted(r["value"]["rectanglelabels"][0] for r in result)
    assert labels == ["Bus", "Car"]


def test_multi_model_predict_with_real_models(client):
    label_config = """
    <View>
      <Image name="image" value="$image"/>
      <RectangleLabels name="label" toName="image" model_score_threshold="0.25"
                      model_path="plate_id_1.pt,container_v12.pt">
        <Label value="container-face"/>
        <Label value="container-front"/>
        <Label value="container-id"/>
        <Label value="truck-front"/>
        <Label value="truck-plate"/>
      </RectangleLabels>
    </View>
    """
    task = {
        "data": {
            "image": "https://s3.amazonaws.com/htx-pub/datasets/mmdetection-ml-test/001bebecea382500.jpg"
        }
    }

    response = client.post(
        "/setup",
        data=json.dumps({"schema": label_config, "project": "42"}),
        content_type="application/json",
    )
    assert response.status_code == 200, response.content

    response = client.post(
        "/predict",
        data=json.dumps({"tasks": [task], "label_config": label_config}),
        content_type="application/json",
    )
    assert response.status_code == 200, response.content

    prediction = response.json["results"][0]
    assert prediction["model_version"] == "yolo"
    assert isinstance(prediction["result"], list)
