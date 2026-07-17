import serial
import time
import random
import numpy as np
import torch
from torchvision import datasets
import zlib
import struct

#  Configuration 
SERIAL_PORT = '/dev/ttyACM0'  # (or something like COM3 on Windows)
BAUD_RATE = 115200

def set_seed(seed=67):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Python built-in random seed
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch random seed for CPU
    torch.manual_seed(seed)

    # PyTorch random seed for all GPUs (if available)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def wait_for_handshake(ser):
    """Blocks until the ESP32 sends the READY signal on startup."""
    print("Waiting for device to be ready...")
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line == "READY":
            ser.write(b"ACK\n")
            return
        elif line:
            print(f"[ESP32 Startup] {line}")

def main_loop():
    print("Loading MNIST dataset into memory...")
    mnist_test = datasets.MNIST(root='./data', train=False, download=True)

    print(f"Opening connection to {SERIAL_PORT}...")

    ser = serial.Serial()
    ser.port = SERIAL_PORT
    ser.baudrate = BAUD_RATE
    ser.timeout = 5.0
    ser.dtr = True
    ser.rts = True
    
    true_predictions = 0
    total_predictions = 0

    try:
        ser.open()
        time.sleep(2)  # A couple moments to stabilize
        
        ser.reset_input_buffer()

        print("\n~~~ Connection Established ~~~")
        
        # Sync with the ESP32
        wait_for_handshake(ser)
        print("Handshake complete. Entering inference loop.")
        
        MAGIC = 0xAA55
        END = 0x55AA
        
        # while True:
        for idx in range(len(mnist_test)):
            #  User confirmation
            # input("\nPress [ENTER] to send a random image, or Ctrl+C to quit...")

            #  Picks a random image
            # idx = random.randint(0, len(mnist_test) - 1)
            image, true_label = mnist_test[idx]
            # Converts to bytes for ser.write()
            image_bytes = np.array(image, dtype=np.uint8).flatten().tobytes()
            
            # Packet structure:
            # [MAGIC: 2 bytes] + [LENGTH: 2 bytes] + [IMAGE DATA: 784 bytes] + [CRC32: 4 bytes] + [END: 2 bytes]
            print(f" Sending Image, Actual Label: {true_label} ")
            packet = (
                struct.pack("<H", MAGIC) +
                struct.pack("<H", len(image_bytes)) +
                image_bytes +
                struct.pack("<I", zlib.crc32(image_bytes, 0)) +
                struct.pack("<H", END)
            )
            # print(packet)

            # Write the packet
            ser.write(packet)
            ser.flush()
            
            #  Read the response from ESP32
            print("Waiting for prediction...")
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if not line:
                    print("[PC] Warning: Serial timeout. No response from ESP32.")
                    break
                
                # prints both the ESP32 info and error logs
                if line.startswith("I (") or line.startswith("E ("):
                    print(f"[ESP32 LOG] {line}")
                    continue

                if "Predicted Digit:" in line:
                    print(f"\n>>> {line} <<<\n")
                    if line[-1] == str(true_label):
                        true_predictions += 1
                        print("Prediction is correct :)")
                    else:
                        print("Prediction is incorrect :(")
                    total_predictions += 1
                    break

    except serial.SerialException as e:
        print(f"\nSerial Error: {e}")
        print("Check if the port is correct and the device is plugged in.")
    except KeyboardInterrupt:
        print("\n\nExiting. Closing serial port...")
        print(f"Total Predictions: {total_predictions}, Correct Predictions: {true_predictions}")
        print(f"Accuracy: {true_predictions / total_predictions * 100:.2f}%")
    finally:
        print(f"Total Predictions: {total_predictions}, Correct Predictions: {true_predictions}")
        print(f"Accuracy: {true_predictions / total_predictions * 100:.2f}%")
        if ser.is_open:
            ser.close()


if __name__ == "__main__":
    set_seed(67)
    main_loop()