import pandas as pd
import numpy as np
from mlflow.pyfunc.model import PythonModel

class SkopsModelWrapper(PythonModel):
    """Simple wrapper for skops-serialized models.

    Provide explicit type hints on `predict` to avoid MLflow data_validation warnings.
    """

    def load_context(self, context) -> None:
        import skops.io as skops_io_local
        self.model = skops_io_local.load(context.artifacts["skops_model"])

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        # model_input llega como DataFrame
        return self.model.predict(model_input)
