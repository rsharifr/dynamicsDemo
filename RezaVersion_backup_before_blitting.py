"""
v = ω × r  Live Classroom Visualizer
======================================
Connects to StepperBLE over Bluetooth and displays the relationship
between angular velocity (ω), radius (r), and linear velocity (v = ω × r).
Includes a text command box to send motor commands (F400, M50, S etc.)
directly to the Arduino over BLE.

INSTALL DEPENDENCIES (run once):
  pip install bleak matplotlib numpy

USAGE:
  1. Power on Arduino, confirm StepperBLE is advertising
  2. Disconnect any other BLE app first
  3. python v_omega_r_visualizer.py
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

# ── Shared state ──────────────────────────────────────────────
omega_buf   = np.zeros(BUFFER_SIZE)
r_buf       = np.zeros(BUFFER_SIZE)
v_buf       = np.zeros(BUFFER_SIZE)
a1_buf      = np.zeros(BUFFER_SIZE)
a2_buf      = np.zeros(BUFFER_SIZE)
a3_buf      = np.zeros(BUFFER_SIZE)
ar_buf      = np.zeros(BUFFER_SIZE)
at_buf      = np.zeros(BUFFER_SIZE)
az_buf      = np.zeros(BUFFER_SIZE)

# Placeholder buffers — not yet computed, wired up in a later step
rdot_buf        = np.zeros(BUFFER_SIZE)
rddot_buf       = np.zeros(BUFFER_SIZE)
theta_buf       = np.zeros(BUFFER_SIZE)
thetadot_buf    = np.zeros(BUFFER_SIZE)
thetaddot_buf   = np.zeros(BUFFER_SIZE)

buf_lock    = threading.Lock()

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
        # Raw readouts from Arduino
        gyro1 = math.radians(vals[0])
        gyro2 = math.radians(vals[1])
        gyro3 = math.radians(vals[2])
        acc1  = vals[3]
        acc2  = vals[4]
        acc3  = vals[5]
        # Gyro magnitude — always positive, represents angular speed
        magnitude = math.sqrt(gyro1**2 + gyro2**2 + gyro3**2)
        # Apply direction sign from 7th value (1 = forward, -1 = reverse)
        direction = float(vals[6]) if len(vals) >= 7 else 1.0
        omega_rads = magnitude * direction
        r_m        = latest_r
        v_ms       = omega_rads * r_m
        with buf_lock:
            global omega_buf, r_buf, v_buf, a1_buf, a2_buf, a3_buf, ar_buf, at_buf, az_buf
            omega_buf   = np.roll(omega_buf,   -1);  omega_buf[-1]   = omega_rads
            r_buf       = np.roll(r_buf,       -1);  r_buf[-1]       = r_m
            v_buf       = np.roll(v_buf,       -1);  v_buf[-1]       = v_ms
            a1_buf      = np.roll(a1_buf,      -1);  a1_buf[-1]      = acc1
            a2_buf      = np.roll(a2_buf,      -1);  a2_buf[-1]      = acc2
            a3_buf      = np.roll(a3_buf,      -1);  a3_buf[-1]      = acc3
            ar_buf = -a1_buf
            at_buf = -a3_buf
            az_buf = -a2_buf
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
    root.title("Polar frame verification")
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

    fig = plt.Figure(figsize=(11, 8), facecolor="#1e1e1e")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.65, wspace=0.38)

    t = np.linspace(-WINDOW_SECONDS, 0, BUFFER_SIZE)

    # 3x3 panel layout — row 1: r, r_dot, r_ddot | row 2: theta, theta_dot, theta_ddot
    # row 3: a_radial, a_transverse, a_z
    # NOTE: rdot/rddot/theta/thetadot/thetaddot are placeholder (zero) buffers —
    # data wiring for these comes in a later step.
    panel_specs = [
        dict(key="r",          buf_name="r_buf",        title="Radius  r  (m)",
             ylabel="r (m)",        unit="m",     fmt=".3f", color="#378ADD", ylim=(0.055, 0.260)),
        dict(key="rdot",       buf_name="rdot_buf",     title="Radial Velocity  ṙ  (m/s)",
             ylabel="ṙ (m/s)",     unit="m/s",   fmt=".3f", color="#1D9E75", ylim=(-1, 1)),
        dict(key="rddot",      buf_name="rddot_buf",    title="Radial Acceleration  r̈  (m/s²)",
             ylabel="r̈ (m/s²)",   unit="m/s²",  fmt=".3f", color="#E24B4A", ylim=(-2, 2)),

        dict(key="theta",      buf_name="theta_buf",     title="Angle  θ  (rad)",
             ylabel="θ (rad)",      unit="rad",   fmt=".2f", color="#378ADD", ylim=(-math.pi, math.pi)),
        dict(key="thetadot",   buf_name="thetadot_buf",  title="Angular Velocity  θ̇  (rad/s)",
             ylabel="θ̇ (rad/s)",  unit="rad/s", fmt=".2f", color="#1D9E75", ylim=(-20, 20)),
        dict(key="thetaddot",  buf_name="thetaddot_buf", title="Angular Acceleration  θ̈  (rad/s²)",
             ylabel="θ̈ (rad/s²)", unit="rad/s²",fmt=".2f", color="#E24B4A", ylim=(-50, 50)),

        dict(key="ar",         buf_name="ar_buf",       title="Radial Acceleration  a_r  (m/s²)",
             ylabel="a_r (m/s²)",   unit="m/s²",  fmt=".2f", color="#E24B4A", ylim=(-10, 10)),
        dict(key="at",         buf_name="at_buf",       title="Transverse Acceleration  a_θ  (m/s²)",
             ylabel="a_θ (m/s²)",   unit="m/s²",  fmt=".2f", color="#378ADD", ylim=(-10, 10)),
        dict(key="az",         buf_name="az_buf",       title="Vertical Acceleration  a_z  (m/s²)",
             ylabel="a_z (m/s²)",   unit="m/s²",  fmt=".2f", color="#1D9E75", ylim=(-10, 10)),
    ]

    axes  = {}
    lines = {}
    texts = {}

    for idx, spec in enumerate(panel_specs):
        row, col = divmod(idx, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#2a2a2a")
        ax.tick_params(colors="#cccccc", labelsize=8)
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#ffffff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.grid(True, alpha=0.2, color="#555555")

        ax.set_title(spec["title"], fontsize=10)
        ax.set_xlabel("time (sec)"); ax.set_ylabel(spec["ylabel"])
        ax.set_xlim(-WINDOW_SECONDS, 0); ax.set_ylim(*spec["ylim"])
        ax.axhline(0, color="#555555", linewidth=0.5)

        line, = ax.plot(t, globals()[spec["buf_name"]], color=spec["color"], linewidth=1.3)
        text  = ax.text(0.02, 0.90, "", transform=ax.transAxes,
                         fontsize=8, color=spec["color"],
                         bbox=dict(boxstyle="round", fc="#2a2a2a", alpha=0.8))

        axes[spec["key"]]  = ax
        lines[spec["key"]] = line
        texts[spec["key"]] = text

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

    # ── Periodic updates ──
    def update_graphs():
        with buf_lock:
            current = {spec["key"]: globals()[spec["buf_name"]].copy() for spec in panel_specs}

        for spec in panel_specs:
            key = spec["key"]
            buf = current[key]
            lines[key].set_ydata(buf)
            texts[key].set_text(f"{spec['buf_name']} = {buf[-1]:{spec['fmt']}} {spec['unit']}")

        rb = current["r"]
        axes["r"].set_ylim(0.03, max(rb.max() * 1.2, 0.24))

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
