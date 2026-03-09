from lambda_folder import lambda_function as lf

sample = {
    "cpu_usage": 50, "ram_usage": 60, "temperature": 70,
    "disk_io": 30, "network_traffic": 100,
    "cpu_spike_count": 1, "ram_spike_count": 0, "uptime_hours": 10
}

event = {"body": sample}
print(lf.lambda_handler(event, None))