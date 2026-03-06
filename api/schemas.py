<<<<<<< HEAD
from pydantic import BaseModel, Field
from typing import List

class ServerMetrics(BaseModel):
    cpu_usage:        float = Field(..., ge=0, le=100)
    ram_usage:        float = Field(..., ge=0, le=100)
    disk_io:          float = Field(..., ge=0, le=100)
    network_traffic:  float = Field(..., ge=0)
    temperature:      float = Field(..., ge=0, le=150)
    cpu_spike_count:  int   = Field(..., ge=0)
    ram_spike_count:  int   = Field(..., ge=0)
    uptime_hours:     float = Field(..., ge=0)

    model_config = {
        "protected_namespaces": (),   # ← fix warning
        "json_schema_extra": {
            "example": {
                "cpu_usage": 87.5,
                "ram_usage": 91.2,
                "disk_io": 78.4,
                "network_traffic": 520.0,
                "temperature": 83.0,
                "cpu_spike_count": 14,
                "ram_spike_count": 11,
                "uptime_hours": 1500.0
            }
        }
    }

class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # ← fix warning

    will_fail:        bool
    probability:      float
    risk_level:       str
    time_to_failure:  str
    top_causes:       List[str]
    model_version:    str

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # ← fix warning

    status:        str
    model_loaded:  bool
    version:       str

class MetricsResponse(BaseModel):
    accuracy:   float
    precision:  float
    recall:     float
    f1:         float
=======
from pydantic import BaseModel, Field
from typing import List

class ServerMetrics(BaseModel):
    cpu_usage:        float = Field(..., ge=0, le=100, description="CPU usage %")
    ram_usage:        float = Field(..., ge=0, le=100, description="RAM usage %")
    disk_io:          float = Field(..., ge=0, le=100, description="Disk I/O MB/s")
    network_traffic:  float = Field(..., ge=0,         description="Network Mbps")
    temperature:      float = Field(..., ge=0, le=150, description="Temperature °C")
    cpu_spike_count:  int   = Field(..., ge=0,         description="CPU spikes last hour")
    ram_spike_count:  int   = Field(..., ge=0,         description="RAM spikes last hour")
    uptime_hours:     float = Field(..., ge=0,         description="Server uptime hours")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cpu_usage": 87.5,
                "ram_usage": 91.2,
                "disk_io": 78.4,
                "network_traffic": 520.0,
                "temperature": 83.0,
                "cpu_spike_count": 14,
                "ram_spike_count": 11,
                "uptime_hours": 1500.0
            }
        }
    }

class PredictionResponse(BaseModel):
    will_fail:        bool
    probability:      float
    risk_level:       str
    time_to_failure:  str
    top_causes:       List[str]
    model_version:    str

class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    version:       str

class MetricsResponse(BaseModel):
    accuracy:   float
    precision:  float
    recall:     float
    f1:         float
>>>>>>> 585b3f1 (Data local commit)
    roc_auc:    float