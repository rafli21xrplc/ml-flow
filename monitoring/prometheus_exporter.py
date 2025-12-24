from prometheus_client import Counter, Gauge, Histogram

SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'CPU usage in percent')
SYSTEM_RAM_USAGE = Gauge('system_ram_usage_percent', 'RAM usage in percent')
SYSTEM_DISK_USAGE = Gauge('system_disk_usage_percent', 'Disk usage in percent')
SYSTEM_NET_SENT   = Gauge('system_net_sent_bytes', 'Network bytes sent')
SYSTEM_NET_RECV   = Gauge('system_net_recv_bytes', 'Network bytes received')

APP_REQUEST_COUNT = Counter('app_request_count', 'Total requests received')
APP_LATENCY       = Histogram('app_latency_seconds', 'Request latency in seconds')
APP_EXCEPTION     = Counter('app_exception_count', 'Total exceptions')
MODEL_PREDICTION  = Counter('model_prediction_output', 'Prediction output class', ['output_class'])
MODEL_CONFIDENCE  = Gauge('model_last_confidence', 'Confidence score of last prediction')