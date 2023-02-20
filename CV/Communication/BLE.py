# !bluepy/bin/python3
# coding: utf-8
# python 3.8
# require python version >= 3.5

"""This script is made for UART connection for BLE device

The script connects to as specified BLE device and receives data through
notification from Tx characteristic from the BLE device with UART service.
This data  is the plotted ona live graph.

Overall we chose to use MQTT for the communication over BLE as it was as quick and simple to implement.
"""

import asyncio
import time
from collections import deque
from typing import Union
from threading import Thread

import bleak  # pip install bleak


async def main():

    # finding the device thought its name
    devices = await bleak.BleakScanner.discover(10)
    for device in devices:
        print(device)
        if device.name == "Arduino":
            adress = device.address

    async with bleak.BleakClient(adress) as client:
        # paired = await client.connect()
        # print(f"connected: {paired}")
        while True:
            val = await client.read_gatt_char("19b10001-e8f2-537e-4f6c-d104768a1214")
            print(val)

            await asyncio.sleep(2)

            await client.write_gatt_char(
                "19b10001-e8f2-537e-4f6c-d104768a1214", b"\x01"
            )

            await asyncio.sleep(2)

            await client.write_gatt_char(
                "19b10001-e8f2-537e-4f6c-d104768a1214", b"\x00"
            )

            await asyncio.sleep(2)
            print("cycle")


async def scan_peripheral(address):
    "scans for services and characteristics and prints them"

    # connecting to the client
    async with bleak.BleakClient(address) as client:

        # printing all the services and characteristics
        services = await client.get_services()
        for service in services:
            print("Service : ", service)

            for char in service.characteristics:
                print("  Characteristic : ", char)

            print()


async def scan_peripheral_for(
    address: Union[str, bleak.backends.device.BLEDevice], uuid
):
    "scans for certain characteristics and returns on object of that characteristic"

    # connecting to the device
    async with bleak.BleakClient(address) as client:

        # iterating over all the characteristics until a match
        services = await client.get_services()
        for service in services:

            for char in service.characteristics:
                if char.uuid[:8] == uuid:
                    print("found", char.uuid)
                    return char

        print("not found")


def start_bleak():
    asyncio.run(main())


if __name__ == "__main__":
    """th = Thread(target=start_bleak)
    th.start()"""
    start_bleak()
    """while True:
        print("hello")"""
