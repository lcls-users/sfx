import signal
import json
import socket
from multiprocessing import shared_memory, Process
import numpy as np
from ipsana import PsanaImg

# Initialize buffer for each process
psana_img_buffer = {}

def get_psana_img(exp, run, access_mode, detector_name):
    """
    Fetches a PsanaImg object for the given parameters, caching the object to avoid redundant initializations.
    """
    key = (exp, run)
    if key not in psana_img_buffer:
        psana_img_buffer[key] = PsanaImg(exp, run, access_mode, detector_name)
    return psana_img_buffer[key]

def worker_process(server_address):
    # Ignore CTRL+C in the worker process
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Create a new socket for the worker
    worker_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    worker_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    worker_socket.bind(server_address)
    worker_socket.listen()

    while True:
        try:
            # Accept a new connection
            connection, client_address = worker_socket.accept()

            # Receive request data
            request_data = connection.recv(4096).decode('utf-8')

            if request_data == "DONE":
                print("Received shutdown signal. Shutting down server.")
                connection.close()
                break

            request_data = json.loads(request_data)
            exp           = request_data.get('exp')
            run           = request_data.get('run')
            access_mode   = request_data.get('access_mode')
            detector_name = request_data.get('detector_name')
            event         = request_data.get('event')
            mode          = request_data.get('mode')

            # Fetch psana image data
            psana_img = get_psana_img(exp, run, access_mode, detector_name)
            data = psana_img.get(event, None, mode)

            # Keep numpy array in a shared memory
            shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
            shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
            shared_array[:] = data

            response_data = json.dumps({
                'name': shm.name,
                'shape': data.shape,
                'dtype': str(data.dtype)
            })

            # Send response with shared memory details
            connection.sendall(response_data.encode('utf-8'))

            # Wait for the client's acknowledgment
            ack = connection.recv(1024).decode('utf-8')
            if ack == "ACK":
                print(f"Shared memory {shm.name} ready to unlink.")
                unlink_shared_memory(shm.name)
            else:
                print("Did not receive proper acknowledgment from client.")

        except Exception as e:
            print(f"Unexpected error: {e}")
            continue
        finally:
            connection.close()

    worker_socket.close()

def unlink_shared_memory(shm_name):
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass

def start_server(address, num_workers):
    # Create a control socket for sending shutdown signals
    control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    control_socket.bind(('localhost', address[1] + 1))
    control_socket.listen()

    # Create and start worker processes
    processes = []
    for _ in range(num_workers):
        p = Process(target=worker_process, args=(address,))
        p.start()
        processes.append(p)

    print(f"Started {num_workers} worker processes.")
    return processes, control_socket

if __name__ == "__main__":
    server_address = ('localhost', 5000)
    num_workers = 4
    print("Starting server ...")
    processes, control_socket = start_server(server_address, num_workers)
    print("Server started")

    try:
        # Wait to complete, join is wait
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Shutdown signal received")

        for _ in range(num_workers):
            # Trigger connection to unblock accept() in workers
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as trigger_socket:
                trigger_socket.connect(('localhost', server_address[1] + 1))
                trigger_socket.sendall("DONE".encode('utf-8'))

        # Wait to complete, join is wait
        for p in processes:
            p.join()
        control_socket.close()
        print("Server shutdown gracefully.")
