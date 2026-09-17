"""
v = ω × r  Live Classroom Visualizer
======================================
Connects to StepperBLE over Bluetooth and displays the relationship
between angular velocity (ω), radius (r), and linear velocity (v = ω × r).
Includes a text command box to send motor commands (F400, M50, S etc.)
directly to the Arduino over BLE.

INSTALL DEPENDENCIES (run once):
  pip install bleak matplotlib numpy opencv-python pillow

USAGE:
  1. Plug in external USB webcam
  2. Power on Arduino, confirm StepperBLE is advertising
  3. Disconnect any other BLE app first
  4. python v_omega_r_visualizer.py

CAMERA NOTE:
  CAMERA_INDEX = 1 for external USB webcam on a laptop with built-in camera.
  Change to 0 if wrong camera opens.
"""

import asyncio
import threading
import math
import queue
import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
import cv2
from bleak import BleakClient, BleakScanner

# ── UUIDs ─────────────────────────────────────────────────────
DEVICE_NAME    = "StepperBLE"
GYRO_UUID      = "19b10002-e8f2-537e-4f6c-d104768a1214"
POSITION_UUID  = "19b10003-e8f2-537e-4f6c-d104768a1214"
MOTOR_CMD_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"

# ── Settings ──────────────────────────────────────────────────
WINDOW_SECONDS = 15
SAMPLE_RATE_HZ = 10
BUFFER_SIZE    = WINDOW_SECONDS * SAMPLE_RATE_HZ
OMEGA_AXIS     = 1       # Y axis from gyro CSV
CAMERA_INDEX   = 1       # 1 = external USB webcam

# ── Shared state ──────────────────────────────────────────────
omega_buf  = np.zeros(BUFFER_SIZE)
r_buf      = np.zeros(BUFFER_SIZE)
v_buf      = np.zeros(BUFFER_SIZE)
ax_buf     = np.zeros(BUFFER_SIZE)
ay_buf     = np.zeros(BUFFER_SIZE)
az_buf     = np.zeros(BUFFER_SIZE)
buf_lock   = threading.Lock()

latest_r   = 0.0
ble_client = None
connected  = False
status_msg = "Scanning for StepperBLE..."
cmd_queue  = queue.Queue()

# ── BLE callbacks ─────────────────────────────────────────────
def on_gyro_notify(sender, data: bytearray):
    global latest_r
    try:
        vals = [float(x) for x in data.decode("utf-8").strip().split(",")]
        if len(vals) < 6:
            return
        # Gyro magnitude — always positive, represents angular speed
        gx_r = math.radians(vals[0])
        gy_r = math.radians(vals[1])
        gz_r = math.radians(vals[2])
        magnitude = math.sqrt(gx_r**2 + gy_r**2 + gz_r**2)
        # Apply direction sign from 7th value (1 = forward, -1 = reverse)
        direction = float(vals[6]) if len(vals) >= 7 else 1.0
        omega_rads = magnitude * direction
        r_m        = latest_r
        v_ms       = omega_rads * r_m
        with buf_lock:
            global omega_buf, r_buf, v_buf, ax_buf, ay_buf, az_buf
            omega_buf = np.roll(omega_buf, -1);  omega_buf[-1] = omega_rads
            r_buf     = np.roll(r_buf,     -1);  r_buf[-1]     = r_m
            v_buf     = np.roll(v_buf,     -1);  v_buf[-1]     = v_ms
            ax_buf    = np.roll(ax_buf,    -1);  ax_buf[-1]    = vals[3]
            ay_buf    = np.roll(ay_buf,    -1);  ay_buf[-1]    = vals[4]
            az_buf    = np.roll(az_buf,    -1);  az_buf[-1]    = vals[5]
    except Exception as e:
        print(f"Gyro parse error: {e}")

def on_position_notify(sender, data: bytearray):
    global latest_r
    try:
        latest_r = float(data.decode("utf-8").strip()) / 1000.0
    except Exception as e:
        print(f"Position parse error: {e}")

# ── BLE async loop ────────────────────────────────────────────
async def ble_main():
    global connected, status_msg, ble_client
    status_msg = f"Scanning for '{DEVICE_NAME}'..."

    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=15.0)
    if device is None:
        status_msg = f"Could not find '{DEVICE_NAME}' — is Arduino powered on?"
        print(status_msg)
        return

    status_msg = "Connecting..."
    print(f"Found {DEVICE_NAME}, connecting...")

    async with BleakClient(device) as client:
        ble_client = client
        connected  = True
        status_msg = "Connected — subscribing..."
        print("Connected!")

        # Subscribe to gyro data
        try:
            await client.start_notify(GYRO_UUID, on_gyro_notify)
            print("Subscribed to gyro.")
        except Exception as e:
            print(f"Gyro subscribe failed: {e}")
            status_msg = f"Gyro subscribe failed: {e}"
            connected = False
            return

        # Subscribe to position data (non-fatal if missing)
        try:
            await client.start_notify(POSITION_UUID, on_position_notify)
            print("Subscribed to position.")
        except Exception as e:
            print(f"Position subscribe failed (non-fatal): {e}")

        status_msg = "Connected to StepperBLE — streaming"
        print("Streaming data...")

        # Hold connection open, draining command queue every 100ms
        while connected:
            while not cmd_queue.empty():
                try:
                    cmd = cmd_queue.get_nowait()
                    await client.write_gatt_char(
                        MOTOR_CMD_UUID, cmd.encode("utf-8"), response=True)
                    print(f"Sent: {cmd}")
                except Exception as e:
                    print(f"Send error: {e}")
            await asyncio.sleep(0.1)

        # Clean up
        for uuid in [GYRO_UUID, POSITION_UUID]:
            try:
                await client.stop_notify(uuid)
            except Exception:
                pass

    ble_client = None
    status_msg = "Disconnected"
    print("Disconnected.")


def ble_thread_fn():
    asyncio.run(ble_main())


def send_ble_command(cmd: str):
    """Put a command in the queue — BLE loop picks it up within 100ms."""
    if not connected:
        print("Not connected — command not sent")
        return
    cmd_queue.put(cmd)
    print(f"Queued: {cmd}")

# ── Main GUI ──────────────────────────────────────────────────
def build_gui():
    global connected

    root = tk.Tk()
    root.title("v = ω × r  |  Live Classroom Demo")
    root.configure(bg="#1e1e1e")
    root.geometry("1280x760")

    # ── Status bar ──
    status_var = tk.StringVar(value="Starting...")
    tk.Label(root, textvariable=status_var, bg="#1e1e1e", fg="#aaaaaa",
             font=("Helvetica", 11)).pack(side=tk.TOP, fill=tk.X, padx=10, pady=(6, 0))

    # ── Main content ──
    content = tk.Frame(root, bg="#1e1e1e")
    content.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    # Left: graphs
    graph_frame = tk.Frame(content, bg="#1e1e1e")
    graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    fig = plt.Figure(figsize=(9, 6.5), facecolor="#1e1e1e")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.38)

    ax_omega = fig.add_subplot(gs[0, 0])
    ax_r     = fig.add_subplot(gs[0, 1])
    ax_v     = fig.add_subplot(gs[1, 0])
    ax_phase = fig.add_subplot(gs[1, 1])

    for ax in [ax_omega, ax_r, ax_v, ax_phase]:
        ax.set_facecolor("#2a2a2a")
        ax.tick_params(colors="#cccccc", labelsize=8)
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#ffffff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.grid(True, alpha=0.2, color="#555555")

    t = np.linspace(-WINDOW_SECONDS, 0, BUFFER_SIZE)

    # Panel 1: ω
    ax_omega.set_title("Angular Velocity  ω  (rad/s)")
    ax_omega.set_xlabel("time (sec)"); ax_omega.set_ylabel("ω (rad/s)")
    ax_omega.set_xlim(-WINDOW_SECONDS, 0); ax_omega.set_ylim(-20, 20)
    ax_omega.axhline(0, color="#555555", linewidth=0.5)
    line_omega, = ax_omega.plot(t, omega_buf, color="#E24B4A", linewidth=1.5)
    omega_text  = ax_omega.text(0.02, 0.92, "", transform=ax_omega.transAxes,
                                fontsize=9, color="#E24B4A",
                                bbox=dict(boxstyle="round", fc="#2a2a2a", alpha=0.8))

    # Panel 2: r
    ax_r.set_title("Radius  r  (m)")
    ax_r.set_xlabel("time (sec)"); ax_r.set_ylabel("r (m)")
    ax_r.set_xlim(-WINDOW_SECONDS, 0); ax_r.set_ylim(0.055, 0.260)
    line_r,    = ax_r.plot(t, r_buf, color="#378ADD", linewidth=1.5)
    r_text      = ax_r.text(0.02, 0.92, "", transform=ax_r.transAxes,
                            fontsize=9, color="#378ADD",
                            bbox=dict(boxstyle="round", fc="#2a2a2a", alpha=0.8))

    # Panel 3: v
    ax_v.set_title("Linear Velocity  v = ω × r  (m/s)")
    ax_v.set_xlabel("time (sec)"); ax_v.set_ylabel("v (m/s)")
    ax_v.set_xlim(-WINDOW_SECONDS, 0); ax_v.set_ylim(-5, 5)
    ax_v.axhline(0, color="#555555", linewidth=0.5)
    line_v,    = ax_v.plot(t, v_buf, color="#1D9E75", linewidth=1.5)
    v_text      = ax_v.text(0.02, 0.92, "", transform=ax_v.transAxes,
                            fontsize=9, color="#1D9E75",
                            bbox=dict(boxstyle="round", fc="#2a2a2a", alpha=0.8))

    # Panel 4: accelerometer
    ax_phase.set_title("Accelerometer  (m/s², includes gravity)")
    ax_phase.set_xlabel("time (sec)"); ax_phase.set_ylabel("m/s²")
    ax_phase.set_xlim(-WINDOW_SECONDS, 0); ax_phase.set_ylim(-20, 20)
    ax_phase.axhline(0, color="#555555", linewidth=0.5)
    line_ax_plot, = ax_phase.plot(t, ax_buf, color="#E24B4A", linewidth=1.2, label="ay")
    line_ay_plot, = ax_phase.plot(t, ay_buf, color="#378ADD", linewidth=1.2, label="ax")
    line_az_plot, = ax_phase.plot(t, az_buf, color="#1D9E75", linewidth=1.2, label="az")
    ax_phase.legend(fontsize=7, loc="upper left",
                    facecolor="#2a2a2a", labelcolor="#cccccc")

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Command input — sits below the graphs
    cmd_frame = tk.Frame(graph_frame, bg="#1e1e1e")
    cmd_frame.pack(fill=tk.X, pady=(4, 4), padx=4)

    tk.Label(cmd_frame, text="Motor command:", bg="#1e1e1e",
             fg="#cccccc", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(0, 6))

    cmd_entry = tk.Entry(cmd_frame, font=("Courier", 13),
                         bg="#2a2a2a", fg="#ffffff",
                         insertbackground="white", relief=tk.FLAT, width=20)
    cmd_entry.pack(side=tk.LEFT, ipady=6, padx=(0, 4))

    def do_send(event=None):
        cmd = cmd_entry.get().strip().upper()
        if cmd:
            send_ble_command(cmd)
            cmd_entry.delete(0, tk.END)

    tk.Button(cmd_frame, text="Send", command=do_send,
              bg="#378ADD", fg="white", relief=tk.FLAT,
              font=("Helvetica", 10, "bold"), padx=12,
              activebackground="#2a7abf").pack(side=tk.LEFT)

    cmd_entry.bind("<Return>", do_send)

    ref = "F<n> Forward  |  R<n> Reverse  |  S Stop  |  M<mm> Actuator  |  Z Zero  |  P Position"
    tk.Label(cmd_frame, text=ref, bg="#1e1e1e", fg="#555555",
             font=("Courier", 8)).pack(side=tk.LEFT, padx=(12, 0))

    # ── Webcam in separate OpenCV window ──────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Warning: could not open camera index {CAMERA_INDEX} — try 0 or 2")

    def webcam_thread_fn():
        cv2.namedWindow("Live Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live Camera", 640, 480)
        while connected:
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cv2.imshow("Live Camera", frame)
            # waitKey(1) is required for OpenCV window to update
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow("Live Camera")
        cap.release()

    webcam_thread = threading.Thread(target=webcam_thread_fn, daemon=True)
    webcam_thread.start()

    # ── Periodic updates ──
    def update_graphs():
        with buf_lock:
            ob  = omega_buf.copy()
            rb  = r_buf.copy()
            vb  = v_buf.copy()
            axb = ax_buf.copy()
            ayb = ay_buf.copy()
            azb = az_buf.copy()

        line_omega.set_ydata(ob)
        line_r.set_ydata(rb)           # metres
        line_v.set_ydata(vb)           # m/s
        line_ax_plot.set_ydata(axb)
        line_ay_plot.set_ydata(ayb)
        line_az_plot.set_ydata(azb)

        om = max(abs(ob.max()), abs(ob.min()), 1.0)
        ax_omega.set_ylim(-om * 1.2, om * 1.2)
        vm = max(abs(vb.max()), abs(vb.min()), 0.01)
        ax_v.set_ylim(-vm * 1.2, vm * 1.2)
        ax_r.set_ylim(0.03, max(rb.max() * 1.2, 0.24))
        a_max = max(abs(axb.max()), abs(ayb.max()), abs(azb.max()), 1.0)
        ax_phase.set_ylim(-a_max * 1.2, a_max * 1.2)

        omega_text.set_text(f"ω = {ob[-1]:.2f} rad/s")
        r_text.set_text(    f"r = {rb[-1]:.3f} m")
        v_text.set_text(    f"v = {vb[-1]:.3f} m/s")

        canvas.draw_idle()
        status_var.set(status_msg)
        root.after(100, update_graphs)

    def on_close():
        global connected
        connected = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Start BLE thread
    threading.Thread(target=ble_thread_fn, daemon=True).start()

    # Start update loops
    root.after(300, update_graphs)
    root.mainloop()

# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    try:
        build_gui()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("\nPress Enter to close...")
