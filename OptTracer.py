
class OptimizationTracer:
    def __init__(self, additional_function_list: list = None):
        self.timestep = 0
        self.tracedata = []
        self.afunlist = []
        if additional_function_list is not None:
            self.afunlist = additional_function_list

    def callback(self, xk):
        self.timestep += 1

    def ResetTimeStep(self):
        self.timestep = 0

    def ResetTraceData(self):
        self.tracedata = 0

    def ResetAdditionalFunctionList(self):
        self.afunlist = []

    def SetAdditionalFunctionList(self, additional_function_list: list):
        self.afunlist = additional_function_list


class Record_Print_Tracer(OptimizationTracer):
    # Append function list required
    def callback(self, xk):
        self.timestep += 1
        self.tracedata.append(self.afunlist[0](xk))
        for index in range(1, len(self.afunlist)):
            print('time step: ' + str(self.timestep) + '    loss function: ' + str(self.afunlist[index](xk)))
