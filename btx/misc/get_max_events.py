from btx.interfaces.ipsana import PsanaImg

def parse_input():
    """
    Parse command line input.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", help="Experiment name.", required=True, type=str)
    parser.add_argument("-r", "--run", help="Run number.", required=True, type=int)
    parser.add_argument(
        "-d",
        "--det_type",
        help="Detector name, e.g epix10k2M or jungfrau4M.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--start_offset",
        help="Run index of first image to be incorporated into iPCA model.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--num_images",
        help="Total number of images to be incorporated into model.",
        required=True,
        type=int,
    )

    return parser.parse_args()

if __name__ == "__main__":
    # Parse input arguments
    args = parse_input()

    # Create PsanaImg object
    psana_img = PsanaImg(args.exp, args.run, 'idx', args.det_type)

    # Get maximum number of events
    max_events = psana_img.__len__()
    print(f"Maximum number of events: {max_events}")
    
    if max_events < args.num_images:
        print("Number of images requested exceeds the maximum number of events.")
        return max_events
    
    return args.num_images

