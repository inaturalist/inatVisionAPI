import datetime
import json
import time


class Logstasher:

    def __init__(self):
        self.log_file = open("log/logstash.log", "a")

    def log_request(self, request, response, g):
        request_duration = round((time.time() - g.request_start_time) * 1000, 6)
        log_data = {
            "@timestamp": datetime.datetime.now().isoformat(),
            "duration": request_duration,
            "clientip": request.access_route[0],
            "path": request.path,
            "method": request.method,
            "qry": request.args.to_dict(),
            "status_code": response.status_code
        }
        if hasattr(g, "image_uuid"):
            log_data["uuid"] = g.image_uuid
        if hasattr(g, "image_size"):
            log_data["image_size"] = g.image_size
        json.dump(log_data, self.log_file)
        self.log_file.write("\n")
        self.log_file.flush()
