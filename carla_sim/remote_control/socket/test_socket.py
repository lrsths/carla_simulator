import socket
import time
import os
import matplotlib.pyplot as plt


def main():
    server_ip = "192.168.0.2"  # Replace with the server's IP address
    server_port = 12345  # Port to connect to

    # Create socket
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((server_ip, server_port))
    print(f"[CONNECTED TO SERVER] {server_ip}:{server_port}")

    list_pic = []

    for pic in os.listdir("output"):
        if pic.endswith("front_camera.png"):
            list_pic.append(pic)

    list_pic.sort()

    for pic in list_pic:
        data = plt.imread(os.path.join('output', pic))
        try:
            # Input message from user

            # Send message to server
            client.send(data)

            # Receive response from server
            response = client.recv(1024).decode('utf-8')
            print(f"[SERVER RESPONSE] {response}")

            # Simulate doing some other work
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n[CLIENT SHUTTING DOWN]")
        finally:
            client.close()


if __name__ == "__main__":
    main()
