#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
Description: Input args parse.
Author: MindX SDK
Create: 2023
History: NA
"""

import argparse
import ipaddress


def check_args(_args):
    host = _args.host
    try:
        ipaddress.IPv4Address(host)
    except ipaddress.AddressValueError as err:
        raise ValueError("the host of the server is not a valid IPv4 address.") from err
    device_id = _args.device_id
    if device_id and device_id < 0:
        raise ValueError("device_id should greater than or equal to 0.")
    if _args.port <= 1000:
        raise ValueError("for safety, port should greater than 1000.")
    

def input_args_parse():
    parser = argparse.ArgumentParser(description="AI Attack Detection")
    parser.add_argument("--host", "-i", default="", help="The host of the server")
    parser.add_argument("--port", "-p", default=8888, type=int, help="The port of the server")
    parser.add_argument("--device_id", "-d", default=0, type=int, help="Device ID")

    args = parser.parse_args()
    check_args(args)

    return args
