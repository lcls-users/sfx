from btx.interfaces.ipsana import PsanaImg

def main(exp,run,det_type,num_images):
    # Create PsanaImg object
    psana_img = PsanaImg(exp, run, 'idx', det_type)

    # Get maximum number of events
    max_events = psana_img.__len__()
    print(f"Maximum number of events: {max_events}")
    
    if max_events < num_images:
        print("Number of images requested exceeds the maximum number of events.")
        return max_events
    
    return num_images

