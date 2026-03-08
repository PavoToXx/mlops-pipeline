import mlflow
import skops.io as skops_io
import pandas as pd
import numpy as np
from mlflow.pyfunc import PythonModel

class SkopsModelWrapper(PythonModel):
    def load_context(self, context) -> None:
        # context.artifacts will contain absolute paths to artifacts
        self._model = skops_io.load(context.artifacts["skops_model"])

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        return self._model.predict(model_input)

# Register model for "models from code" workflow
mlflow.models.set_model(SkopsModelWrapper())
