import psana

class LaserState:

    def __init__(self, exp, run, det_type, event_on):
        self.exp = exp
        self.run = run
        self.det_type = det_type
        self.ds = psana.DataSource(f'exp={exp}:run={run}')
        self.ecr_det = psana.Detector('evr1')
        self.event_on = event_on

    def is_on(self, evt):
        """Get laser state on/off.
        This function returns True whenever the event code matches the input.
        """
        is_on = False
        ec = self.ecr_det.eventCodes(evt)
        if self.event_on in ec:
            is_on = True
        return is_on

laser_state = LaserState('cxilz0720', 120, 'evr1', 183)
for idx,evt in enumerate(laser_state.ds.events()):
  if laser_state.is_on(evt):
    print(f'event {idx}: laser ON')
  else:
    print(f'event {idx}: laser OFF')
  if (idx > 50 ): break
