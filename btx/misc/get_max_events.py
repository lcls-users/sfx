from btx.interfaces.ipsana import PsanaImg

def main(exp,run,det_type):
    # Create PsanaImg object
    psana_img = PsanaImg(exp, run, 'idx', det_type)

    # Get maximum number of events
    max_events = psana_img.__len__()
    
    return max_events

