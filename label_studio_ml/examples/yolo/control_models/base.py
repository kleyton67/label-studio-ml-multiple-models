import gc
import os
import logging

from pydantic import BaseModel
from typing import Optional, List, Dict, ClassVar
from ultralytics import YOLO
import torch

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import DATA_UNDEFINED_NAME
from label_studio_sdk.label_interface.control_tags import ControlTag
from label_studio_sdk.label_interface import LabelInterface


# use matplotlib plots for debug
DEBUG_PLOT = os.getenv("DEBUG_PLOT", "false").lower() in ["1", "true"]
MODEL_SCORE_THRESHOLD = float(os.getenv("MODEL_SCORE_THRESHOLD", 0.5))
DEFAULT_MODEL_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_ROOT = os.getenv("MODEL_ROOT", DEFAULT_MODEL_ROOT)
os.makedirs(MODEL_ROOT, exist_ok=True)
# if true, allow to use custom model path from the control tag in the labeling config
ALLOW_CUSTOM_MODEL_PATH = os.getenv("ALLOW_CUSTOM_MODEL_PATH", "true").lower() in [
    "1",
    "true",
]

# Global cache for YOLO models
_model_cache = {}
logger = logging.getLogger(__name__)


def get_bool(attr, attr_name, default="false"):
    return attr.get(attr_name, default).lower() in ["1", "true", "yes"]


def parse_model_paths(model_path) -> List[str]:
    """Parse the `model_path` control tag attribute into a list of model filenames.

    A single control tag can point to several models separated by commas, e.g.:
        <RectangleLabels model_path="plate_id_1.pt, container_v12.pt">
        => ["plate_id_1.pt", "container_v12.pt"]

    Whitespace around each entry is ignored and empty entries are dropped, so a
    plain single-model path is returned as a one-element list.
    """
    if not model_path:
        return []
    return [part.strip() for part in str(model_path).split(",") if part.strip()]


class ControlModel(BaseModel):
    """
    Represents a control tag in Label Studio, which is associated with a specific type of labeling task
    and is used to generate predictions using a YOLO model.

    Attributes:
        type (str): Type of the control, e.g., RectangleLabels, Choices, etc.
        control (ControlTag): The actual control element from the Label Studio configuration.
        from_name (str): The name of the control tag, used to link the control to the data.
        to_name (str): The name of the data field that this control is associated with.
        value (str): The value name from the object that this control operates on, e.g., an image or text field.
        model (object): The model instance (e.g., YOLO) used to generate predictions for this control.
        model_paths (List[str]): Model filenames used by this control, parsed from the
            comma-separated `model_path` attribute. A single-model control has one entry.
        label_maps (Dict[str, Dict[str, str]]): Per-model mapping of model labels to Label
            Studio labels, keyed by model filename. Each model keeps its own mapping because
            different models may expose different class names.
        model_path (str): Path to the default YOLO model file (class level).
        model_score_threshold (float): Threshold for prediction scores; predictions below this value will be ignored.
        label_map (Optional[Dict[str, str]]): Union of all per-model label maps, kept for
            validation and backward compatibility. Predictions use `label_maps` per model.
    """

    type: ClassVar[str]
    control: ControlTag
    from_name: str
    to_name: str
    value: str
    model: YOLO
    model_paths: List[str] = []
    label_maps: Dict[str, Dict[str, str]] = {}
    model_path: ClassVar[str]
    model_score_threshold: float = 0.5
    label_map: Optional[Dict[str, str]] = {}
    label_studio_ml_backend: LabelStudioMLBase
    project_id: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def is_control_matched(cls, control) -> bool:
        """Check if the control tag matches the model type.
        Args:
            control (ControlTag): The control tag from the Label Studio Interface.
        """
        raise NotImplementedError("This method should be overridden in derived classes")

    @staticmethod
    def get_from_name_for_label_map(
        label_interface: LabelInterface, target_name: str
    ) -> str:
        """Get the 'from_name' attribute for the label map building."""
        return target_name

    @classmethod
    def create(cls, mlbackend: LabelStudioMLBase, control: ControlTag):
        """Factory method to create an instance of a specific control model class.
        Args:
            mlbackend (LabelStudioMLBase): The ML backend instance.
            control (ControlTag): The control tag from the Label Studio Interface.
        """
        from_name = control.name
        to_name = control.to_name[0]
        value = control.objects[0].value_name

        # if skip is true, don't process this control
        if get_bool(control.attr, "model_skip", "false"):
            logger.info(
                f"Skipping control tag '{control.tag}' with name '{from_name}', model_skip=true found"
            )
            return None
        # read threshold attribute from the control tag, e.g.: <RectangleLabels model_score_threshold="0.5">
        model_score_threshold = float(
            control.attr.get("model_score_threshold")
            or control.attr.get(
                "score_threshold"
            )  # not recommended option, use `model_score_threshold`
            or MODEL_SCORE_THRESHOLD
        )
        model_path_str = (
            ALLOW_CUSTOM_MODEL_PATH and control.attr.get("model_path")
        ) or cls.model_path
        model_paths = parse_model_paths(model_path_str)

        # from_name for label mapping can be differed from control.name (e.g. VideoRectangle)
        label_map_from_name = cls.get_from_name_for_label_map(
            mlbackend.label_interface, from_name
        )

        # Load each model to build its own label map. Models stay cached on the CPU;
        # only one of them is moved to the GPU at a time during prediction.
        label_maps = {}
        model = None
        for path in model_paths:
            yolo_model = cls.get_cached_model(path)
            label_maps[path] = mlbackend.build_label_map(
                label_map_from_name, yolo_model.names.values()
            )
            if model is None:
                model = yolo_model

        # `label_map` keeps the union of all per-model maps: it is used to validate
        # that the control tag has at least one matching label with the model(s).
        label_map = {}
        for path in model_paths:
            label_map.update(label_maps.get(path, {}))

        return cls(
            control=control,
            from_name=from_name,
            to_name=to_name,
            value=value,
            model=model,
            model_paths=model_paths,
            label_maps=label_maps,
            model_score_threshold=model_score_threshold,
            label_map=label_map,
            label_studio_ml_backend=mlbackend,
            project_id=mlbackend.project_id,
        )

    @classmethod
    def load_yolo_model(cls, filename) -> YOLO:
        """Load YOLO model from the file."""
        path = os.path.join(MODEL_ROOT, filename)
        logger.info(f"Loading yolo model: {path}")
        model = YOLO(path)
        logger.info(f"Model {path} names:\n{model.names}")
        return model

    @classmethod
    def get_cached_model(cls, path: str) -> YOLO:
        if path not in _model_cache:
            _model_cache[path] = cls.load_yolo_model(path)
        return _model_cache[path]

    @classmethod
    def offload_model(cls, model: YOLO):
        """Release the GPU memory held by the model's predictor.

        Ultralytics keeps the inference weights inside `model.predictor` (a device
        copy) while `model.model` stays on the CPU. Dropping the predictor frees the
        GPU, but the CPU weights stay cached so the next prediction reuses them
        without reloading the file from disk.
        """
        try:
            model.predictor = None
            model.model.cpu()
        except Exception as e:
            logger.debug(f"Failed to offload model from GPU: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def debug_plot(self, image):
        if not DEBUG_PLOT:
            return

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.imshow(image[..., ::-1])
        plt.axis("off")
        plt.title(self.type)
        plt.show()

    def predict_regions(self, path) -> List[Dict]:
        """Predict regions in the image using the YOLO model.
        Args:
            path (str): Path to the file with media
        """
        raise NotImplementedError("This method should be overridden in derived classes")

    def predict_regions_for_tasks(self, paths: List[str]) -> List[List[Dict]]:
        """Predict regions for a batch of task paths, one model at a time.

        Multi-model control tags run each model over the whole batch and offload it
        before the next model runs, so only one model occupies the GPU at any moment
        (the models may be too large to fit together in GPU memory).
        """
        if len(self.model_paths) <= 1:
            return [self.predict_regions(path) for path in paths]

        results: List[List[Dict]] = [[] for _ in paths]
        for model_path in self.model_paths:
            model = self.get_cached_model(model_path)
            self.model = model
            self.label_map = self.label_maps.get(model_path, {})
            try:
                for i, path in enumerate(paths):
                    results[i].extend(self.predict_regions(path))
            finally:
                self.offload_model(model)
        return results

    def fit(self, event, data, **kwargs):
        """Fit the model."""
        logger.warning("The fit method is not implemented for this control model")
        return False

    def get_path(self, task):
        task_path = task["data"].get(self.value) or task["data"].get(
            DATA_UNDEFINED_NAME
        )
        if task_path is None:
            raise ValueError(
                f"Can't load path using key '{self.value}' from task {task}"
            )
        if not isinstance(task_path, str):
            raise ValueError(f"Path should be a string, but got {task_path}")

        # try path as local file or try to load it from Label Studio instance/download via http
        path = (
            task_path
            if os.path.exists(task_path)
            else self.label_studio_ml_backend.get_local_path(task_path, task_id=task.get("id"))
        )
        logger.debug(f"load_image: {task_path} => {path}")
        return path

    def __str__(self):
        """Return a string with full representation of the control tag."""
        return (
            f"{self.type} from_name={self.from_name}, "
            f"label_map={self.label_map}, model_score_threshold={self.model_score_threshold}"
        )

    class Config:
        arbitrary_types_allowed = True
        protected_namespaces = ("__.*__", "_.*")  # Excludes 'model_'
