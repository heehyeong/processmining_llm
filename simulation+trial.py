import sax.core.process_mining.process_mining as pm
from config import SAX4BPM_CONFIG

event_log = pm.import_xes(SAX4BPM_CONFIG['data_file'])
baseline_pm = pm.discover_bpmn_model(event_log)

with open("output_model.bpmn", "w", encoding="utf-8") as f:
    f.write(baseline_pm)