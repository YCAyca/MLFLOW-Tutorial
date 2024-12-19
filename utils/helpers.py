import mlflow

import psutil
import time
import mlflow
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlShutdown

import psutil
import time
import csv
import matplotlib.pyplot as plt
from pynvml import nvmlDeviceGetUtilizationRates
import mlflow


def log_system_metrics(interval=1, gpu_handle=None):
    """
    Logs system metrics (CPU, memory, GPU) over time with step increments.
    :param interval: Interval in seconds to log metrics.
    :param gpu_handle: GPU handle for logging GPU metrics (optional).
    """
    step = 0  # Start the step counter
    try:
        while True:
            # Log CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            mlflow.log_metric("cpu_usage", cpu_percent, step=step)

            # Log memory metrics
            memory_info = psutil.virtual_memory()
            mlflow.log_metric("memory_usage_percent", memory_info.percent, step=step)
            mlflow.log_metric("memory_used_mb", memory_info.used / (1024 * 1024), step=step)
            mlflow.log_metric("memory_available_mb", memory_info.available / (1024 * 1024), step=step)

            # Log GPU metrics if GPU handle is available
            if gpu_handle:
                gpu_utilization = nvmlDeviceGetUtilizationRates(gpu_handle)
                mlflow.log_metric("gpu_usage", gpu_utilization.gpu, step=step)
                mlflow.log_metric("gpu_memory_usage", gpu_utilization.memory, step=step)

            # Sleep for the interval
            time.sleep(interval)
            step += 1  # Increment the step for the next log
    except KeyboardInterrupt:
        print("System metrics logging interrupted.")



def log_dict_as_params(config, prefix=""):
    """
    Log a dictionary as MLflow parameters.
    :param config: Dictionary to log.
    :param prefix: Prefix to add to parameter names for nested dictionaries.
    """
    for key, value in config.items():
        if isinstance(value, dict):
            # Recursively log nested dictionaries
            log_dict_as_params(value, prefix=f"{prefix}{key}.")
        else:
            # Log the parameter
            mlflow.log_param(f"{prefix}{key}", value)