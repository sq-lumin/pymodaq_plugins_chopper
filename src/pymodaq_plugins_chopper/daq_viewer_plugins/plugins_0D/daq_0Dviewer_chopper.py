from ctypes import c_ulong

import numpy as np
from easydict import EasyDict as edict
from pymodaq.utils.daq_utils import ThreadCommand, getLineInfo
from pymodaq.utils.data import DataFromPlugins, Axis
from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main

from pymodaq_plugins_daqmx.hardware.national_instruments.daqmx import DAQmx, DAQ_analog_types, DAQ_thermocouples,\
    DAQ_termination, Edge, DAQ_NIDAQ_source, \
    ClockSettings, AIChannel, Counter, AIThermoChannel, AOChannel, TriggerSettings, DOChannel, DIChannel

logger = set_logger(get_module_name(__file__))


class DAQ_0DViewer_chopper(DAQ_Viewer_base):
    """Specific DAQmx plugin for getting analog input data as 0D or 1D data

    """
    params = comon_parameters+[{'title' : 'Chopper Params', 'name' : 'chopper_params', 'type' : 'group', 'children' : [
            {'title': 'Display type:', 'name': 'display', 'type': 'list', 'limits': ['0D', '1D'], 'value' : '1D'},
            {'title': 'Frequency Acq.:', 'name': 'frequency', 'type': 'int', 'value': 250000, 'min': 1},
            {'title': 'Nsamples:', 'name': 'Nsamples', 'type': 'int', 'value': 50, 'default': 50, 'min': 1},
            {'title' : 'Trigger Source', 'name': 'trigger_source', 'type': 'list', 'limits': DAQmx.get_NIDAQ_channels(), 'value' : '/Dev2/PFI0'},
            {'title': 'AI:', 'name': 'ai_channel', 'type': 'list',
             'limits': DAQmx.get_NIDAQ_channels(source_type='Analog_Input'),
             'value': DAQmx.get_NIDAQ_channels(devices = [DAQmx.get_NIDAQ_devices()[1]], source_type='Analog_Input')[1]}, #Dev2
            ]}
        ]
    hardware_averaging = True
    live_mode_available = True

    def __init__(self, parent=None, params_state=None):
        super().__init__(parent, params_state)

        self.channels = {}
        self.clock_settings = {}
        self.trigger_settings = {}
        self.data_tot = None
        self.live = False
        self.Naverage = 1
        self.ind_average = 0
            
    def commit_settings(self, param):
        """
        """

        self.update_tasks()

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object) custom object of a PyMoDAQ plugin (Slave case). None if only one detector by controller (Master case)

        Returns
        -------
        self.status (edict): with initialization status: three fields:
            * info (str)
            * controller (object) initialized controller
            *initialized: (bool): False if initialization failed otherwise True
        """
        if controller is None:
            controller = {}
        new_controller = controller.copy()
        new_controller['chopper'] = DAQmx()
        self.ini_detector_init(old_controller=controller, new_controller=new_controller)

        self.update_tasks()

        info = "Current measurement ready"
        initialized = True

        return info, initialized

    def update_tasks(self):

        self.channels = [AIChannel(name=self.settings.child('chopper_params').child('ai_channel').value(),
                                      source='Analog_Input', analog_type='Voltage',
                                      value_min=-10., value_max=10., termination='Diff', ),
                            ]

        self.clock_settings = ClockSettings(frequency=self.settings.child('chopper_params').child('frequency').value(),
                                               Nsamples=self.settings.child('chopper_params').child('Nsamples').value(), repetition=False)  #repetition = self.live
        #print(self.settings['chopper_params']['Nsamples'])
        self.trigger_settings = TriggerSettings(trig_source=self.settings.child('chopper_params').child('trigger_source').value(), enable=True, edge=Edge.names()[0], level=1)    #Rising Edge
        self.controller['chopper'].update_task(self.channels, self.clock_settings, self.trigger_settings)
        #set the proper buffer size manually. Not needed if Nsamples is big enough (>=50), but needed for smaller Nsamples.
        self.controller['chopper']._task.SetBufInputBufSize(c_ulong(self.settings.child('chopper_params').child('Nsamples').value()))
        
    def close(self):
        """
        Terminate the communication protocol
        """
        pass
        ##

    def grab_data(self, Naverage=1, **kwargs):
        """

        Parameters
        ----------
        Naverage: (int) Number of hardware averaging
        kwargs: (dict) of others optionals arguments
        """
        

        update = False

        if 'live' in kwargs:
            if kwargs['live'] != self.live:
                update = True
            self.live = kwargs['live']

        if Naverage != self.Naverage:
            self.Naverage = Naverage
            update = True

        if update:
            self.update_tasks()

        self.ind_average = 0
        self.data_tot = np.zeros((len(self.channels), self.clock_settings.Nsamples))

        while not self.controller['chopper'].isTaskDone():
            self.stop()
        self.read_chopper()

    
    def read_chopper(self, callback = None):
        if callback is None:
            callback = self.read_data
        if self.controller['chopper'].c_callback is None:
            self.controller['chopper'].register_callback(callback, 'Nsamples', self.clock_settings.Nsamples)
        self.controller['chopper'].task.StartTask()
        self._taskRunning = True
        
    def read_data(self, taskhandle, status, samples=0, callbackdata=None):
        #print(f'going to read {self.clock_settings_ai.Nsamples} samples, callbakc {samples}')
        data = self.controller['chopper'].readAnalog(len(self.channels), self.clock_settings)
        if not self.live:
            self.stop()
        self.ind_average += 1
        self.data_tot += 1 / self.Naverage * data
        if self.ind_average == self.Naverage:
            self.emit_data(self.data_tot)
            self.ind_average = 0
            self.data_tot = np.zeros((len(self.channels), self.clock_settings.Nsamples))

        return 0  #mandatory for the PyDAQmx callback

    def emit_data(self, data):
        channels_name = [ch.name for ch in self.channels]

        if self.settings.child('chopper_params').child('display').value() == '0D':
            data = np.mean(data, 1)

        #data = np.squeeze(data)
        # print(f'shape is {data.shape}')
        # print(data)
        if len(self.channels) == 1 and data.size == 1:
            data_export = [np.array([data[0]])]
        else:
            data_export = [np.squeeze(data[ind, :]) for ind in range(len(self.channels))]

        # print(f'data len is {len(data_export)} and shape is {data_export[0].shape}')
        self.data_grabed_signal.emit([DataFromPlugins(
            name='NI AI',
            data=data_export,
            dim=f'Data{self.settings.child("chopper_params").child("display").value()}', labels=channels_name)])

    def stop(self):
        try:
            self.controller['chopper'].task.StopTask()
            self._taskRunning = False
            self.emit_status(ThreadCommand('Update_Status', ['Chopper plugin stopped']))
        except:
            pass
        ##############################

        return ''
    
    def get_controller_class(self):
        return DAQmx
    
if __name__ == '__main__':
    main(__file__)