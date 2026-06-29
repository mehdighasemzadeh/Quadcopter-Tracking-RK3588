#!/usr/bin/env python3
"""
Server that receives 3 float velocity commands (Vx, Vy, Vz) via TCP and
controls a PX4/Gazebo quadcopter in OFFBOARD mode.

Usage:
  1. Start PX4 SITL (Gazebo) – default MAVLink UDP port 14540.
  2. Run this script.
  3. Send velocity packets from another machine / script.

Dependencies: mavsdk, struct, asyncio
  pip install mavsdk
"""

import asyncio
import struct
import socket
import signal
import sys
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityNedYaw)

# ------------------------------------------------------------------------------
# Configuration (adjust to your setup)
# ------------------------------------------------------------------------------
TCP_HOST = '0.0.0.0'          # Listen on all interfaces
TCP_PORT = 5005              # Must match the sender's Gazebo_sim_port
DRONE_ADDRESS = "udp://:14540"  # Default MAVSDK connection for PX4 SITL

TAKEOFF_ALTITUDE = 10.0       # metres
VELOCITY_STREAM_HZ = 50      # Setpoint sending rate (Hz)

# ------------------------------------------------------------------------------
# Global velocity setpoint (Vx, Vy, Vz) in NED frame (m/s), updated by TCP
# ------------------------------------------------------------------------------
current_velocity = (0.0, 0.0, 0.0)
vel_lock = asyncio.Lock()

# ------------------------------------------------------------------------------
# TCP data reader
# ------------------------------------------------------------------------------
async def tcp_velocity_receiver():
    """Accept a TCP connection and continuously unpack 3 floats."""
    global current_velocity

    # Set up listening socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TCP_HOST, TCP_PORT))
    server.listen(1)
    server.setblocking(False)

    print(f"TCP server listening on {TCP_HOST}:{TCP_PORT}")

    while True:
        # Accept a client (non‑blocking, wrapped by asyncio)
        conn, addr = await asyncio.get_event_loop().sock_accept(server)
        print(f"TCP client connected: {addr}")
        conn.setblocking(False)

        # Read exactly 12 bytes per packet (3 floats * 4 bytes)
        data = b''
        try:
            while True:
                # Read chunk
                chunk = await asyncio.get_event_loop().sock_recv(conn, 12 - len(data))
                if not chunk:
                    print("TCP client disconnected")
                    conn.close()
                    break
                data += chunk

                # When we have a full 12‑byte packet, unpack it
                while len(data) >= 12:
                    packet = data[:12]
                    data = data[12:]
                    try:
                        vx, vy, vz = struct.unpack('!3f', packet)
                        async with vel_lock:
                            current_velocity = (vx, vy, vz)
                        # Optional print for debugging (uncomment if needed)
                        # print(f"\rGot velocity: {vx:+6.2f}, {vy:+6.2f}, {vz:+6.2f} m/s", end='')
                    except struct.error:
                        # Corrupted packet – discard and resync
                        print("Warning: bad packet, discarding")
                        data = b''  # reset buffer
                        break

        except (ConnectionResetError, ConnectionAbortedError):
            print("TCP client disconnected")
            conn.close()
        except Exception as e:
            print(f"TCP error: {e}")
            conn.close()

# ------------------------------------------------------------------------------
# Drone control
# ------------------------------------------------------------------------------
async def run_drone():
    """Connect to the drone, take off, switch to offboard velocity control."""
    global current_velocity

    drone = System()
    await drone.connect(system_address=DRONE_ADDRESS)

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected!")
            break

    # Wait for global position estimate
    print("Waiting for global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("Global position estimate OK")
            break

    # Arm the drone
    print("Arming...")
    await drone.action.arm()
    print("Armed")

    # Take off to desired altitude
    print(f"Taking off to {TAKEOFF_ALTITUDE} m...")
    await drone.action.set_takeoff_altitude(TAKEOFF_ALTITUDE)
    await drone.action.takeoff()

    # Wait until altitude is reached (using position NED telemetry)
    async for pos in drone.telemetry.position_velocity_ned():
        if pos.position.down_m * -1 >= TAKEOFF_ALTITUDE * 0.95:  # NED: down is negative
            print("Reached target altitude")
            break

    # Initialise the offboard setpoint with zero velocity and start offboard mode
    print("Starting OFFBOARD mode with velocity control...")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Offboard start failed: {error}")
        await drone.action.disarm()
        return

    print("Offboard mode active. Listening for velocity commands...")

    # Control loop: send setpoints at fixed rate
    interval = 1.0 / VELOCITY_STREAM_HZ
    next_time = asyncio.get_event_loop().time()

    try:
        while True:
            # Retrieve latest velocity command
            async with vel_lock:
                vx, vy, vz = current_velocity

            # Send velocity NED (yaw rate = 0.0, let the drone hold yaw)
            await drone.offboard.set_velocity_ned(
                VelocityNedYaw(vx, vy, vz, 0.0)
            )

            # Maintain the setpoint rate
            next_time += interval
            await asyncio.sleep(max(0, next_time - asyncio.get_event_loop().time()))

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Stop offboard and land / disarm safely
        print("Stopping offboard mode...")
        try:
            await drone.offboard.stop()
        except OffboardError:
            pass
        print("Landing...")
        await drone.action.land()
        # Wait for landing detection (optional)
        async for in_air in drone.telemetry.in_air():
            if not in_air:
                break
        print("Disarming...")
        await drone.action.disarm()

# ------------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------------
async def main():
    # Run TCP receiver and drone controller concurrently
    tcp_task = asyncio.create_task(tcp_velocity_receiver())
    drone_task = asyncio.create_task(run_drone())

    # Wait for either task to finish (drone_task will exit on Ctrl+C)
    done, pending = await asyncio.wait(
        [tcp_task, drone_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    # Cancel remaining tasks
    for task in pending:
        task.cancel()
    # Allow graceful cleanup
    await asyncio.gather(*pending, return_exceptions=True)

if __name__ == "__main__":
    # Handle Ctrl+C properly
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")
