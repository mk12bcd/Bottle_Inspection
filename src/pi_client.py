import socket

pc_ip = "192.168.1.1"

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((pc_ip, 5000))

client.send("Hello from Pi".encode())

response = client.recv(1024).decode()
print("PC says:", response)