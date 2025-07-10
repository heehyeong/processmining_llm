import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import sax
from sax.core.process_data.formatters.xes_formatter import XESFormatter
from sax.core.process_data.formatters.csv_formatter import CSVFormatter
from sax.core.utils.constants import Constants
from lingam.utils import make_dot
from sax.core.process_data.tabular_data import TabularEventData
from pm4py.objects.conversion.log import converter as log_converter
import sax.core.process_mining.process_mining as pm
import sax.core.causal_process_discovery.causal_discovery as cd

#Import event log file and create event_log data object
fileName = "Road Traffic Fine Management Process_1_all/Road_Traffic_Fine_Management_Process.xes.gz"
event_log = pm.import_xes(fileName,timestamp_format="%Y-%m-%d %H:%M:%S.%f")
event_log.getData()

print('Mandatory Properties of the parsed event log: \n',event_log.getMandatoryProperties())
print('Optional properties of the parsed event log: \n',event_log.getOptionalProperties())

# Run heuristic process miner on the whole dataset and visualize the resulting net
net = pm.discover_heuristics_net(event_log)
pm.view_heuristics_net(net)

result = cd.discover_causal_dependencies(event_log)
graph = cd.view_causal_dependencies(result)
graph.view()