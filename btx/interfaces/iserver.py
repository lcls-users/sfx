import signal
import json
import socket
from multiprocessing import shared_memory, Process, Queue
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

def worker_process(connection_queue):
    # Ignore CTRL+C in the worker process
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:
        try:
            connection, client_address = connection_queue.get()
            if connection is None:
                print("Worker received shutdown signal.")
                break

            # Receive request data
            request_data = connection.recv(4096).decode('utf-8')

            if request_data == "DONE":
                print("Received shutdown signal. Shutting down worker.")
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
            connection.close()
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

def unlink_shared_memory(shm_name):
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass

def dispatcher_process(server_socket, connection_queue):
    while True:
        try:
            connection, client_address = server_socket.accept()
            connection_queue.put((connection, client_address))
        except Exception as e:
            print(f"Dispatcher error: {e}")
            break

def start_server(address, num_workers):
    # Init TCP socket, set reuse, bind, and listen for connections
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(address)
    server_socket.listen()

    connection_queue = Queue()

    # Create and start dispatcher process
    dispatcher = Process(target=dispatcher_process, args=(server_socket, connection_queue))
    dispatcher.start()

    # Create and start worker processes
    processes = []
    for _ in range(num_workers):
        p = Process(target=worker_process, args=(connection_queue,))
        p.start()
        processes.append(p)

    print(f"Started {num_workers} worker processes and dispatcher.")
    return processes, dispatcher, server_socket, connection_queue

if __name__ == "__main__":
    server_address = ('localhost', 5000)
    num_workers = 4
    print("Starting server ...")
    processes, dispatcher, server_socket, connection_queue = start_server(server_address, num_workers)
    print("Server started")

    try:
        # Wait to complete, join is wait
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Shutdown signal received")

        for _ in range(len(processes)):
            connection_queue.put((None, None))

        dispatcher.terminate()
        dispatcher.join()

        for p in processes:
            p.join()
        server_socket.close()
        print("Server shutdown gracefully.")
