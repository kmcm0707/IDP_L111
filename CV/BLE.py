# !bluepy/bin/python3
# coding: utf-8

# using python 3.8
# require python version >= 3.5
"""This script is made for UART connection for BLE device

The script connects to as specified BLE device and receives data through
notification from Tx characteristic from the BLE device with UART service.
This data  is the plotted ona live graph.
"""
import asyncio
import time
from collections import deque
from typing import Union
from threading import Thread

import bleak  # pip install bleak
from matplotlib import pyplot as plt
import numpy as np  # pip install numpy


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
    exit()
    device = await bleak.BleakScanner.find_device_by_filter(
        lambda d, ad: d.name and d.name == "pi"
    )

    """ This is because mac connects to peripheral devices using uuid rather then MAC address,
        and this uuid is generated by the computer so not always consistent """

    " use the bellow for finding the device by address "
    # device = await bleak.BleakScanner.find_device_by_address(address)

    # device will be None if device was not found
    if device is None:
        print("device not found")
        return

    # locating the characteristic starting with 6e400002
    # which indicates Tx characteristic in UART service
    char = await scan_peripheral_for(device, "6e400002")

    # connecting to the device and enabling notification
    graph = await test_notify(device, char)
    timestamp = time.strftime("%Y-%m-%d_%X")

    with open("data/data_{timestamp}.csv", "w") as f:
        f.write(graph.cvs_dump())


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


async def test_notify(address, char_uuid):
    """This function connects to the device and enable notification
    the data received is handled by the graph.notify_handler method"""

    # connecting
    async with bleak.BleakClient(address) as client:
        print("connected")

        # creating an graph object and passing the method as handler
        # to access the data later and plot the graph
        graph = LiveGraph()

        # this enables notification
        await client.start_notify(char_uuid, graph.notify_handler)
        print("notify enabled")

        # waiting for 20 seconds
        await asyncio.sleep(20)

        # stops receiving notification
        await client.stop_notify(char_uuid)

        graph.plot()
        plt.show()

        return graph

        # print(graph.data_collected)


def start_bleak():
    asyncio.run(main())


if __name__ == "__main__":
    """th = Thread(target=start_bleak)
    th.start()"""
    start_bleak()
    """while True:
        print("hello")"""