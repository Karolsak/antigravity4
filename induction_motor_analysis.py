"""Advanced Electrical Engineering Analysis Suite (Tkinter).

Practical focus: synchronous generator short-circuit study with extended tabs for
faults, protection, control, thermal, economic, harmonic, and comprehensive
engineering analysis.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


EPS = 1e-9


# =============================================================================
# Core machine model and calculations
# =============================================================================


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def generator_base_values(mva: float, kv_ll: float) -> tuple[float, float, float]:
    """Return (I_base[A], Z_base[ohm], V_phase[V])."""
    s_va = mva * 1e6
    v_ll = kv_ll * 1e3
    i_base = s_va / (np.sqrt(3) * v_ll)
    z_base = (v_ll ** 2) / s_va
    v_phase = v_ll / np.sqrt(3)
    return i_base, z_base, v_phase


def induced_voltage_pu_from_pf_unity(xd_pu: float, p_pu: float = 1.0) -> float:
    """At unity PF, I=1∠0 pu and E = V + j Xd I where V=1∠0 pu."""
    return float(np.sqrt(1.0 + (xd_pu * p_pu) ** 2))


def short_circuit_currents_pu(e0_pu: float, xd_tr_pu: float, xd_sync_pu: float) -> tuple[float, float]:
    """Return (initial transient/sc subtransient approximation, final steady)."""
    initial = e0_pu / max(xd_tr_pu, EPS)
    final = e0_pu / max(xd_sync_pu, EPS)
    return initial, final


def short_circuit_waveform(
    t: np.ndarray,
    f_hz: float,
    i_init_rms: float,
    i_final_rms: float,
    t_transient: float,
    dc_tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate asymmetrical fault current and its RMS envelope."""
    env = i_final_rms + (i_init_rms - i_final_rms) * np.exp(-t / max(t_transient, EPS))
    ac = np.sqrt(2.0) * env * np.sin(2.0 * np.pi * f_hz * t)
    dc = np.sqrt(2.0) * i_init_rms * np.exp(-t / max(dc_tau, EPS))
    total = ac + dc
    return total, env


def speed_dynamics_sim(
    total_time: float,
    dt: float,
    speed_ref_rpm: float,
    step_time: float,
    load_before: float,
    load_after: float,
    inertia: float,
    damping: float,
    torque_limit: float,
    kp: float,
    ki: float,
    kd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simple rotor dynamic model with PID and fuzzy-like controller."""
    t = np.arange(0.0, total_time + dt, dt)
    w_ref = speed_ref_rpm * 2.0 * np.pi / 60.0

    w_pid = np.zeros_like(t)
    w_fuzzy = np.zeros_like(t)
    torque_pid = np.zeros_like(t)
    torque_fuzzy = np.zeros_like(t)

    integ = 0.0
    prev_e = 0.0

    def fuzzy_ctrl(e_rpm: float, de_rpm: float) -> float:
        if e_rpm > 150:
            return 0.95 * torque_limit
        if e_rpm > 60:
            return 0.7 * torque_limit if de_rpm > -5 else 0.55 * torque_limit
        if e_rpm > 10:
            return 0.45 * torque_limit
        if e_rpm > -10:
            return 0.3 * torque_limit
        if e_rpm > -40:
            return 0.18 * torque_limit
        return 0.05 * torque_limit

    for k in range(1, len(t)):
        tl = load_after if t[k] >= step_time else load_before

        e = w_ref - w_pid[k - 1]
        integ = clamp(integ + e * dt, -2e4, 2e4)
        dedt = (e - prev_e) / dt
        u_pid = clamp(kp * e + ki * integ + kd * dedt, 0.0, torque_limit)
        prev_e = e

        dwpid = (u_pid - tl - damping * w_pid[k - 1]) / max(inertia, EPS)
        w_pid[k] = max(0.0, w_pid[k - 1] + dwpid * dt)
        torque_pid[k] = u_pid

        e_f = (w_ref - w_fuzzy[k - 1]) * 60.0 / (2.0 * np.pi)
        de_f = ((w_ref - w_fuzzy[k - 1]) - (w_ref - w_fuzzy[k - 2] if k > 1 else w_ref)) * 60.0 / (2.0 * np.pi * dt)
        u_f = clamp(fuzzy_ctrl(e_f, de_f), 0.0, torque_limit)

        dwf = (u_f - tl - damping * w_fuzzy[k - 1]) / max(inertia, EPS)
        w_fuzzy[k] = max(0.0, w_fuzzy[k - 1] + dwf * dt)
        torque_fuzzy[k] = u_f

    return t, w_pid * 60.0 / (2.0 * np.pi), w_fuzzy * 60.0 / (2.0 * np.pi), torque_pid


# =============================================================================
# Tkinter application
# =============================================================================


class AdvancedPowerSystemApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Advanced Electrical Engineering Studio")
        self.geometry("1450x920")
        self.minsize(1200, 760)

        # Generator scenario defaults from user question
        self.mva = tk.DoubleVar(value=250.0)
        self.kv = tk.DoubleVar(value=25.0)
        self.freq = tk.DoubleVar(value=60.0)
        self.xd = tk.DoubleVar(value=1.6)
        self.xd_prime = tk.DoubleVar(value=0.23)
        self.p_pu = tk.DoubleVar(value=1.0)

        self.t_transient = tk.DoubleVar(value=0.18)
        self.dc_tau = tk.DoubleVar(value=0.06)

        self.therm_amb = tk.DoubleVar(value=25.0)
        self.therm_rth = tk.DoubleVar(value=0.015)
        self.therm_cth = tk.DoubleVar(value=75000.0)

        self.energy_price = tk.DoubleVar(value=0.11)
        self.maint_cost = tk.DoubleVar(value=9000.0)
        self.capacity_factor = tk.DoubleVar(value=0.72)

        self.h5 = tk.DoubleVar(value=4.0)
        self.h7 = tk.DoubleVar(value=2.5)
        self.h11 = tk.DoubleVar(value=1.2)

        # Q13/Q14: synchronous generator with resistive load
        self.gen_e0 = tk.DoubleVar(value=3000.0)
        self.gen_xs = tk.DoubleVar(value=6.0)
        self.gen_r = tk.DoubleVar(value=8.0)

        self._build_menu()
        self._build_notebook()
        self._build_status()

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.after(100, self.update_all)

    # ---------------------------------------------------------------------
    # UI setup
    # ---------------------------------------------------------------------
    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        self.config(menu=menu)

        file_m = tk.Menu(menu, tearoff=0)
        file_m.add_command(label="Recalculate", command=self.update_all)
        file_m.add_separator()
        file_m.add_command(label="Exit", command=self.destroy)
        menu.add_cascade(label="File", menu=file_m)

        help_m = tk.Menu(menu, tearoff=0)
        help_m.add_command(label="About", command=lambda: messagebox.showinfo(
            "About",
            "Advanced Electrical Engineering Studio\n"
            "Synchronous generator + system-level analysis tabs\n"
            "Includes fault, protection, controls, thermal/economic/harmonic studies."
        ))
        menu.add_cascade(label="Help", menu=help_m)

    def _build_status(self) -> None:
        self.status_var = tk.StringVar(value="Ready")
        lbl = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w")
        lbl.grid(row=2, column=0, sticky="ew", padx=3, pady=3)

    def _build_notebook(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.nb = ttk.Notebook(self)
        self.nb.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.tab_input = ttk.Frame(self.nb)
        self.tab_fault = ttk.Frame(self.nb)
        self.tab_protection = ttk.Frame(self.nb)
        self.tab_speed = ttk.Frame(self.nb)
        self.tab_thermal = ttk.Frame(self.nb)
        self.tab_economic = ttk.Frame(self.nb)
        self.tab_harmonic = ttk.Frame(self.nb)
        self.tab_advanced = ttk.Frame(self.nb)
        self.tab_gen_load = ttk.Frame(self.nb)

        self.nb.add(self.tab_input, text="Main Menu / Inputs")
        self.nb.add(self.tab_fault, text="Modelling & Fault Current")
        self.nb.add(self.tab_protection, text="Protection Coordination")
        self.nb.add(self.tab_speed, text="Speed Controller")
        self.nb.add(self.tab_thermal, text="Thermal Analysis")
        self.nb.add(self.tab_economic, text="Economic Analysis")
        self.nb.add(self.tab_harmonic, text="Harmonic & Power Quality")
        self.nb.add(self.tab_advanced, text="Comprehensive Analysis")
        self.nb.add(self.tab_gen_load, text="Q13/Q14: Generator Load")

        self._build_tab_input()
        self._build_tab_fault()
        self._build_tab_protection()
        self._build_tab_speed()
        self._build_tab_thermal()
        self._build_tab_economic()
        self._build_tab_harmonic()
        self._build_tab_advanced()
        self._build_tab_gen_load()

    # ---------------------------------------------------------------------
    # Tab 1: Inputs + detailed result text
    # ---------------------------------------------------------------------
    def _build_tab_input(self) -> None:
        t = self.tab_input
        t.columnconfigure(1, weight=1)
        t.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(t, text="Input Parameters (with sliders)")
        left.grid(row=0, column=0, sticky="ns", padx=6, pady=6)

        params = [
            ("Power rating S (MVA)", self.mva, 50.0, 1200.0),
            ("Line voltage (kV)", self.kv, 6.6, 36.0),
            ("Frequency (Hz)", self.freq, 50.0, 60.0),
            ("Synchronous reactance Xd (pu)", self.xd, 0.5, 3.0),
            ("Transient reactance X'd (pu)", self.xd_prime, 0.08, 0.6),
            ("Active power P (pu)", self.p_pu, 0.2, 1.2),
            ("Transient decay T' (s)", self.t_transient, 0.05, 1.0),
            ("DC offset tau (s)", self.dc_tau, 0.01, 0.2),
        ]
        for r, (name, var, lo, hi) in enumerate(params):
            ttk.Label(left, text=name).grid(row=r, column=0, sticky="w", padx=4, pady=3)
            value_lbl = ttk.Label(left, width=10, text=f"{var.get():.3f}")
            value_lbl.grid(row=r, column=2, sticky="e")
            ttk.Scale(
                left,
                from_=lo,
                to=hi,
                variable=var,
                orient="horizontal",
                length=240,
                command=lambda _v, vv=var, ll=value_lbl: (ll.config(text=f"{vv.get():.3f}"), self.update_all()),
            ).grid(row=r, column=1, padx=4)

        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=len(params), column=0, columnspan=3, pady=8)
        ttk.Button(btn_frame, text="Start", command=self.update_all).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Stop", command=lambda: self.status_var.set("Stopped (manual).")).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Reset", command=self._reset_defaults).pack(side="left", padx=4)

        right = ttk.LabelFrame(t, text="Detailed Engineering Explanation")
        right.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.result_text = tk.Text(right, wrap="word", font=("Consolas", 10))
        self.result_text.grid(row=0, column=0, sticky="nsew")
        s = ttk.Scrollbar(right, orient="vertical", command=self.result_text.yview)
        s.grid(row=0, column=1, sticky="ns")
        self.result_text.configure(yscrollcommand=s.set)

    # ---------------------------------------------------------------------
    # Tab 2: fault modelling and simulation
    # ---------------------------------------------------------------------
    def _build_tab_fault(self) -> None:
        t = self.tab_fault
        t.columnconfigure(0, weight=1)
        t.rowconfigure(0, weight=1)

        self.fault_fig = Figure(figsize=(10, 6), dpi=100)
        self.fault_canvas = FigureCanvasTkAgg(self.fault_fig, master=t)
        self.fault_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        ctrl = ttk.Frame(t)
        ctrl.grid(row=1, column=0, sticky="ew")
        ttk.Button(ctrl, text="Start", command=self.update_fault_plot).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Stop", command=lambda: self.status_var.set("Fault plot frozen.")).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Reset", command=self.update_fault_plot).pack(side="left", padx=4)

    # ---------------------------------------------------------------------
    # Tab 3: protection coordination
    # ---------------------------------------------------------------------
    def _build_tab_protection(self) -> None:
        t = self.tab_protection
        t.columnconfigure(0, weight=1)
        t.rowconfigure(0, weight=1)

        self.prot_fig = Figure(figsize=(10, 6), dpi=100)
        self.prot_canvas = FigureCanvasTkAgg(self.prot_fig, master=t)
        self.prot_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        panel = ttk.Frame(t)
        panel.grid(row=1, column=0, sticky="ew")
        self.ip_pri = tk.DoubleVar(value=1.2)
        self.tms_pri = tk.DoubleVar(value=0.12)
        self.ip_back = tk.DoubleVar(value=1.05)
        self.tms_back = tk.DoubleVar(value=0.33)

        items = [("Primary Ip", self.ip_pri, 0.8, 2.0), ("Primary TMS", self.tms_pri, 0.05, 1.0),
                 ("Backup Ip", self.ip_back, 0.8, 2.0), ("Backup TMS", self.tms_back, 0.05, 1.2)]
        for c, (n, v, lo, hi) in enumerate(items):
            f = ttk.Frame(panel)
            f.grid(row=0, column=c, padx=6)
            ttk.Label(f, text=n).pack()
            lbl = ttk.Label(f, text=f"{v.get():.2f}")
            lbl.pack()
            ttk.Scale(f, from_=lo, to=hi, variable=v, length=140,
                      command=lambda _x, vv=v, ll=lbl: (ll.config(text=f"{vv.get():.2f}"), self.update_protection_plot())).pack()

        ttk.Button(panel, text="Start", command=self.update_protection_plot).grid(row=0, column=4, padx=4)
        ttk.Button(panel, text="Stop", command=lambda: self.status_var.set("Protection plot frozen.")).grid(row=0, column=5, padx=4)
        ttk.Button(panel, text="Reset", command=self.update_protection_plot).grid(row=0, column=6, padx=4)

    # ---------------------------------------------------------------------
    # Tab 4: speed control
    # ---------------------------------------------------------------------
    def _build_tab_speed(self) -> None:
        t = self.tab_speed
        t.columnconfigure(0, weight=1)
        t.rowconfigure(0, weight=1)

        self.speed_fig = Figure(figsize=(10, 6), dpi=100)
        self.speed_canvas = FigureCanvasTkAgg(self.speed_fig, master=t)
        self.speed_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.kp = tk.DoubleVar(value=4.0)
        self.ki = tk.DoubleVar(value=2.3)
        self.kd = tk.DoubleVar(value=0.04)
        self.ref_rpm = tk.DoubleVar(value=3600.0)
        self.step_time = tk.DoubleVar(value=3.0)
        self.load_before = tk.DoubleVar(value=20.0)
        self.load_after = tk.DoubleVar(value=65.0)
        self.inertia = tk.DoubleVar(value=12.0)
        self.damping = tk.DoubleVar(value=0.02)

        panel = ttk.Frame(t)
        panel.grid(row=1, column=0, sticky="ew")
        ctrl_values = [
            ("Kp", self.kp, 0.0, 20.0),
            ("Ki", self.ki, 0.0, 10.0),
            ("Kd", self.kd, 0.0, 1.0),
            ("Ref rpm", self.ref_rpm, 1000.0, 4200.0),
            ("Step t (s)", self.step_time, 0.3, 8.0),
            ("Load1 Nm", self.load_before, 0.0, 120.0),
            ("Load2 Nm", self.load_after, 0.0, 120.0),
        ]
        for c, (n, v, lo, hi) in enumerate(ctrl_values):
            f = ttk.Frame(panel)
            f.grid(row=0, column=c, padx=5)
            ttk.Label(f, text=n).pack()
            lbl = ttk.Label(f, text=f"{v.get():.2f}")
            lbl.pack()
            ttk.Scale(f, from_=lo, to=hi, variable=v, length=115,
                      command=lambda _x, vv=v, ll=lbl: (ll.config(text=f"{vv.get():.2f}"), self.update_speed_plot())).pack()

        ttk.Button(panel, text="Start", command=self.update_speed_plot).grid(row=0, column=8, padx=4)
        ttk.Button(panel, text="Stop", command=lambda: self.status_var.set("Speed simulation paused.")).grid(row=0, column=9, padx=4)
        ttk.Button(panel, text="Reset", command=self.update_speed_plot).grid(row=0, column=10, padx=4)

    # ---------------------------------------------------------------------
    # Tab 5: thermal
    # ---------------------------------------------------------------------
    def _build_tab_thermal(self) -> None:
        t = self.tab_thermal
        t.columnconfigure(0, weight=1)
        t.rowconfigure(0, weight=1)

        self.thermal_fig = Figure(figsize=(10, 6), dpi=100)
        self.thermal_canvas = FigureCanvasTkAgg(self.thermal_fig, master=t)
        self.thermal_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        panel = ttk.Frame(t)
        panel.grid(row=1, column=0, sticky="ew")
        params = [("Tamb °C", self.therm_amb, 0.0, 55.0), ("Rth °C/W", self.therm_rth, 0.005, 0.06),
                  ("Cth J/°C", self.therm_cth, 20000.0, 200000.0)]
        for c, (n, v, lo, hi) in enumerate(params):
            f = ttk.Frame(panel)
            f.grid(row=0, column=c, padx=8)
            ttk.Label(f, text=n).pack()
            lbl = ttk.Label(f, text=f"{v.get():.3f}")
            lbl.pack()
            ttk.Scale(f, from_=lo, to=hi, variable=v, length=160,
                      command=lambda _x, vv=v, ll=lbl: (ll.config(text=f"{vv.get():.3f}"), self.update_thermal_plot())).pack()
        ttk.Button(panel, text="Start", command=self.update_thermal_plot).grid(row=0, column=4)
        ttk.Button(panel, text="Stop", command=lambda: self.status_var.set("Thermal plot frozen.")).grid(row=0, column=5)
        ttk.Button(panel, text="Reset", command=self.update_thermal_plot).grid(row=0, column=6)

    # ---------------------------------------------------------------------
    # Tab 6: economic
    # ---------------------------------------------------------------------
    def _build_tab_economic(self) -> None:
        t = self.tab_economic
        t.columnconfigure(0, weight=1)
        t.rowconfigure(0, weight=1)

        self.econ_fig = Figure(figsize=(10, 6), dpi=100)
        self.econ_canvas = FigureCanvasTkAgg(self.econ_fig, master=t)
        self.econ_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        panel = ttk.Frame(t)
        panel.grid(row=1, column=0, sticky="ew")
        params = [("$/kWh", self.energy_price, 0.03, 0.30), ("Annual maintenance $", self.maint_cost, 1000.0, 40000.0),
                  ("Capacity factor", self.capacity_factor, 0.2, 1.0)]
        for c, (n, v, lo, hi) in enumerate(params):
            f = ttk.Frame(panel)
            f.grid(row=0, column=c, padx=8)
            ttk.Label(f, text=n).pack()
            lbl = ttk.Label(f, text=f"{v.get():.3f}")
            lbl.pack()
            ttk.Scale(f, from_=lo, to=hi, variable=v, length=160,
                      command=lambda _x, vv=v, ll=lbl: (ll.config(text=f"{vv.get():.3f}"), self.update_econ_plot())).pack()

        ttk.Button(panel, text="Start", command=self.update_econ_plot).grid(row=0, column=4)
        ttk.Button(panel, text="Stop", command=lambda: self.status_var.set("Economic plot frozen.")).grid(row=0, column=5)
        ttk.Button(panel, text="Reset", command=self.update_econ_plot).grid(row=0, column=6)

    # ---------------------------------------------------------------------
    # Tab 7: harmonics
    # ---------------------------------------------------------------------
    def _build_tab_harmonic(self) -> None:
        t = self.tab_harmonic
        t.columnconfigure(0, weight=1)
        t.rowconfigure(0, weight=1)

        self.harm_fig = Figure(figsize=(10, 6), dpi=100)
        self.harm_canvas = FigureCanvasTkAgg(self.harm_fig, master=t)
        self.harm_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        panel = ttk.Frame(t)
        panel.grid(row=1, column=0, sticky="ew")
        params = [("5th %", self.h5, 0.0, 20.0), ("7th %", self.h7, 0.0, 20.0), ("11th %", self.h11, 0.0, 20.0)]
        for c, (n, v, lo, hi) in enumerate(params):
            f = ttk.Frame(panel)
            f.grid(row=0, column=c, padx=8)
            ttk.Label(f, text=n).pack()
            lbl = ttk.Label(f, text=f"{v.get():.2f}")
            lbl.pack()
            ttk.Scale(f, from_=lo, to=hi, variable=v, length=170,
                      command=lambda _x, vv=v, ll=lbl: (ll.config(text=f"{vv.get():.2f}"), self.update_harm_plot())).pack()
        ttk.Button(panel, text="Start", command=self.update_harm_plot).grid(row=0, column=4)
        ttk.Button(panel, text="Stop", command=lambda: self.status_var.set("Harmonic plot frozen.")).grid(row=0, column=5)
        ttk.Button(panel, text="Reset", command=self.update_harm_plot).grid(row=0, column=6)

    # ---------------------------------------------------------------------
    # Tab 8: comprehensive
    # ---------------------------------------------------------------------
    def _build_tab_advanced(self) -> None:
        t = self.tab_advanced
        t.columnconfigure(0, weight=1)
        t.rowconfigure(0, weight=1)

        self.adv_fig = Figure(figsize=(12, 8), dpi=100)
        self.adv_canvas = FigureCanvasTkAgg(self.adv_fig, master=t)
        self.adv_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        panel = ttk.Frame(t)
        panel.grid(row=1, column=0, sticky="ew")
        ttk.Button(panel, text="Start", command=self.update_advanced_plot).pack(side="left", padx=4)
        ttk.Button(panel, text="Stop", command=lambda: self.status_var.set("Comprehensive tab paused.")).pack(side="left", padx=4)
        ttk.Button(panel, text="Reset", command=self.update_advanced_plot).pack(side="left", padx=4)

    # ---------------------------------------------------------------------
    # Update methods
    # ---------------------------------------------------------------------
    def _compute_case(self) -> dict[str, float]:
        i_base, z_base, v_phase = generator_base_values(self.mva.get(), self.kv.get())
        e0_pu = induced_voltage_pu_from_pf_unity(self.xd.get(), self.p_pu.get())
        i_init_pu, i_final_pu = short_circuit_currents_pu(e0_pu, self.xd_prime.get(), self.xd.get())

        return {
            "i_base": i_base,
            "z_base": z_base,
            "v_phase": v_phase,
            "e0_pu": e0_pu,
            "i_init_pu": i_init_pu,
            "i_final_pu": i_final_pu,
            "i_init_a": i_init_pu * i_base,
            "i_final_a": i_final_pu * i_base,
        }

    def update_all(self) -> None:
        data = self._compute_case()
        self._update_main_text(data)
        self.update_fault_plot()
        self.update_protection_plot()
        self.update_speed_plot()
        self.update_thermal_plot()
        self.update_econ_plot()
        self.update_harm_plot()
        self.update_advanced_plot()
        self.update_gen_load_plot()
        self.status_var.set(
            f"Solved: E0={data['e0_pu']:.3f} pu, I_initial={data['i_init_a']:.0f} A, I_final={data['i_final_a']:.0f} A"
        )

    def _update_main_text(self, d: dict[str, float]) -> None:
        txt = []
        txt.append("SYNCHRONOUS GENERATOR FAULT STUDY (Detailed)\n")
        txt.append("Given: 250 MVA, 25 kV, 3-phase, unity PF, Xd=1.6 pu, X'd=0.23 pu.\n")
        txt.append("\n1) Base values:\n")
        txt.append(f"   I_base = S/(√3V) = {d['i_base']:.2f} A\n")
        txt.append(f"   Z_base = V^2/S = {d['z_base']:.4f} Ω\n")
        txt.append("\n2) Pre-fault induced emf E0:\n")
        txt.append("   At unity PF: I=1∠0 pu, V=1∠0 pu\n")
        txt.append("   E0 = V + jXdI = 1 + j1.6\n")
        txt.append(f"   |E0| = sqrt(1^2 + 1.6^2) = {d['e0_pu']:.4f} pu\n")
        txt.append("\n3) Short-circuit currents:\n")
        txt.append("   Initial symmetrical (using X'd): I0 = E0/X'd\n")
        txt.append(f"   I0 = {d['i_init_pu']:.4f} pu = {d['i_init_a']:.1f} A\n")
        txt.append("   Final steady (if breaker fails, using Xd): If = E0/Xd\n")
        txt.append(f"   If = {d['i_final_pu']:.4f} pu = {d['i_final_a']:.1f} A\n")
        txt.append("\n4) Differential equation used in simulation:\n")
        txt.append("   dIenv/dt = -(Ienv - Ifinal)/T' ; fault AC current i_ac(t)=√2*Ienv*sin(ωt).\n")
        txt.append("   Total asymmetrical current adds DC offset: i(t)=i_ac + √2*I0*e^{-t/τ}.\n")
        txt.append("\nModel correctness checks:\n")
        txt.append(" • t=0 envelope equals initial current.\n")
        txt.append(" • As t→∞ envelope converges to final steady current.\n")
        txt.append(" • All dynamic states are clamped to prevent overflow/non-physical values.\n")

        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", "".join(txt))

    def update_fault_plot(self) -> None:
        d = self._compute_case()
        t = np.linspace(0.0, 0.4, 2200)
        i_total, env = short_circuit_waveform(
            t,
            self.freq.get(),
            d["i_init_a"],
            d["i_final_a"],
            self.t_transient.get(),
            self.dc_tau.get(),
        )

        self.fault_fig.clear()
        ax1 = self.fault_fig.add_subplot(211)
        ax2 = self.fault_fig.add_subplot(212)

        ax1.plot(t, i_total, lw=1.0, color="tab:red", label="Asymmetrical fault current")
        ax1.plot(t, np.sqrt(2.0) * env, "--", color="tab:blue", label="+AC envelope")
        ax1.plot(t, -np.sqrt(2.0) * env, "--", color="tab:blue")
        ax1.grid(alpha=0.3)
        ax1.set_ylabel("Instantaneous Current (A)")
        ax1.legend(fontsize=8)

        ax2.plot(t, env, color="tab:green", lw=2.0, label="RMS envelope")
        ax2.axhline(d["i_init_a"], color="tab:orange", ls=":", label="Initial RMS")
        ax2.axhline(d["i_final_a"], color="tab:purple", ls=":", label="Final RMS")
        ax2.grid(alpha=0.3)
        ax2.set_ylabel("RMS Current (A)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(fontsize=8)

        self.fault_fig.tight_layout()
        self.fault_canvas.draw_idle()

    def update_protection_plot(self) -> None:
        self.prot_fig.clear()
        ax = self.prot_fig.add_subplot(111)

        i_mult = np.logspace(0.01, 2.0, 400)

        def iec_si(i, ip, tms):
            m = np.maximum(i / max(ip, EPS), 1.0001)
            return tms * 0.14 / (np.power(m, 0.02) - 1.0)

        tp = iec_si(i_mult, self.ip_pri.get(), self.tms_pri.get())
        tb = iec_si(i_mult, self.ip_back.get(), self.tms_back.get())

        ax.loglog(i_mult, tp, lw=2, label="Primary relay")
        ax.loglog(i_mult, tb, lw=2, label="Backup relay")
        ax.fill_between(i_mult, tp, tb, where=(tb > tp), alpha=0.2, label="Coordination margin")
        ax.axhline(0.3, ls=":", color="tab:red", label="Recommended CTI 0.3 s")

        ax.set_title("Protection Coordination (IEC Standard Inverse)")
        ax.set_xlabel("Current Multiple (I/Ip)")
        ax.set_ylabel("Trip Time (s)")
        ax.grid(which="both", alpha=0.3)
        ax.legend(fontsize=8)
        self.prot_fig.tight_layout()
        self.prot_canvas.draw_idle()

    def update_speed_plot(self) -> None:
        t, pid_rpm, fz_rpm, torque_pid = speed_dynamics_sim(
            total_time=10.0,
            dt=0.02,
            speed_ref_rpm=self.ref_rpm.get(),
            step_time=self.step_time.get(),
            load_before=self.load_before.get(),
            load_after=self.load_after.get(),
            inertia=self.inertia.get(),
            damping=self.damping.get(),
            torque_limit=120.0,
            kp=self.kp.get(),
            ki=self.ki.get(),
            kd=self.kd.get(),
        )
        ref = np.full_like(t, self.ref_rpm.get())

        self.speed_fig.clear()
        ax1 = self.speed_fig.add_subplot(211)
        ax2 = self.speed_fig.add_subplot(212)

        ax1.plot(t, pid_rpm, lw=2, label="PID")
        ax1.plot(t, fz_rpm, lw=2, label="Fuzzy")
        ax1.plot(t, ref, "k--", lw=1.2, label="Reference")
        ax1.axvline(self.step_time.get(), ls=":", color="tab:red", label="Step load")
        ax1.set_ylabel("Speed (rpm)")
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=8)

        ax2.plot(t, torque_pid, color="tab:green", lw=2, label="PID torque command")
        ax2.set_ylabel("Command Torque (N·m)")
        ax2.set_xlabel("Time (s)")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)
        self.speed_fig.tight_layout()
        self.speed_canvas.draw_idle()

    def update_thermal_plot(self) -> None:
        d = self._compute_case()
        copper_loss = 3.0 * (d["i_final_a"] ** 2) * 0.004
        t = np.linspace(0, 6 * 3600, 1800)
        tau = max(self.therm_rth.get() * self.therm_cth.get(), EPS)
        t_inf = self.therm_amb.get() + copper_loss * self.therm_rth.get()
        temp = t_inf - (t_inf - self.therm_amb.get()) * np.exp(-t / tau)

        self.thermal_fig.clear()
        ax = self.thermal_fig.add_subplot(111)
        ax.plot(t / 3600.0, temp, lw=2, label="Winding temperature")
        ax.axhline(105.0, ls="--", color="tab:red", label="Class-B advisory line")
        ax.set_title("Thermal Model: Cth·dT/dt = Ploss - (T-Tamb)/Rth")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        self.thermal_fig.tight_layout()
        self.thermal_canvas.draw_idle()

    def update_econ_plot(self) -> None:
        years = np.arange(1, 21)
        annual_mwh = self.mva.get() * self.capacity_factor.get() * 8760
        revenue = annual_mwh * 1000.0 * self.energy_price.get()
        opex = self.maint_cost.get() + 0.008 * revenue
        cash = revenue - opex
        disc = 0.08
        npv = np.cumsum(cash / np.power(1.0 + disc, years))

        self.econ_fig.clear()
        ax1 = self.econ_fig.add_subplot(211)
        ax2 = self.econ_fig.add_subplot(212)
        ax1.bar(years, np.full_like(years, cash) / 1e6, color="tab:blue")
        ax1.set_ylabel("Annual net cash (M$)")
        ax1.grid(alpha=0.25)

        ax2.plot(years, npv / 1e6, marker="o", lw=2, color="tab:green")
        ax2.set_ylabel("Cumulative NPV (M$)")
        ax2.set_xlabel("Year")
        ax2.grid(alpha=0.25)

        self.econ_fig.tight_layout()
        self.econ_canvas.draw_idle()

    def update_harm_plot(self) -> None:
        harmonics = np.array([1, 5, 7, 11])
        mags = np.array([100.0, self.h5.get(), self.h7.get(), self.h11.get()])
        thd = np.sqrt(np.sum(mags[1:] ** 2)) / max(mags[0], EPS) * 100.0

        self.harm_fig.clear()
        ax1 = self.harm_fig.add_subplot(211)
        ax2 = self.harm_fig.add_subplot(212)

        ax1.bar(harmonics, mags, width=0.8, color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
        ax1.set_xticks(harmonics)
        ax1.set_ylabel("Magnitude (% of fundamental)")
        ax1.set_title("Harmonic Spectrum")
        ax1.grid(alpha=0.25)

        qual = ["THD (%)", "IEEE-519 limit\n(typ. 5%)"]
        vals = [thd, 5.0]
        ax2.bar(qual, vals, color=["tab:purple", "tab:gray"])
        ax2.set_ylabel("Percent")
        ax2.set_title(f"Power Quality Indicator: THD = {thd:.2f}%")
        ax2.grid(alpha=0.25)

        self.harm_fig.tight_layout()
        self.harm_canvas.draw_idle()

    def update_advanced_plot(self) -> None:
        d = self._compute_case()
        self.adv_fig.clear()

        ax1 = self.adv_fig.add_subplot(221)
        ax2 = self.adv_fig.add_subplot(222)
        ax3 = self.adv_fig.add_subplot(223)
        ax4 = self.adv_fig.add_subplot(224)

        # A) Current metrics
        ax1.bar(["Initial SC kA", "Final SC kA"], [d["i_init_a"] / 1000.0, d["i_final_a"] / 1000.0],
                color=["tab:red", "tab:blue"])
        ax1.set_title("Fault Current Metrics")
        ax1.grid(alpha=0.25)

        # B) Stability margin heuristic
        margin = (self.xd.get() - self.xd_prime.get()) / max(self.xd.get(), EPS)
        ax2.pie([margin, 1 - margin], labels=["Transient reserve", "Used"], autopct="%1.1f%%")
        ax2.set_title("Reactance Separation Index")

        # C) Protection margin
        cti = self.tms_back.get() - self.tms_pri.get()
        ax3.bar(["CTI proxy"], [cti], color="tab:green" if cti >= 0.2 else "tab:orange")
        ax3.axhline(0.2, color="tab:red", ls=":")
        ax3.set_ylabel("Seconds (proxy)")
        ax3.set_title("Protection Coordination Health")
        ax3.grid(alpha=0.25)

        # D) composite score
        thd = np.sqrt(self.h5.get() ** 2 + self.h7.get() ** 2 + self.h11.get() ** 2)
        score = 100.0
        score -= clamp((thd - 5.0) * 4.0, 0.0, 25.0)
        score -= clamp((d["i_init_pu"] - 5.0) * 3.0, 0.0, 30.0)
        score -= clamp((0.2 - cti) * 80.0, 0.0, 25.0)
        score = clamp(score, 0.0, 100.0)
        ax4.bar(["Overall index"], [score], color="tab:blue")
        ax4.axhline(80, color="tab:green", ls="--")
        ax4.set_ylim(0, 100)
        ax4.set_title("Comprehensive Engineering Index")
        ax4.grid(alpha=0.25)

        self.adv_fig.tight_layout()
        self.adv_canvas.draw_idle()

    # ---------------------------------------------------------------------
    # Tab 9: Q13/Q14 – 3-phase generator with resistive load
    # ---------------------------------------------------------------------
    def _build_tab_gen_load(self) -> None:
        t = self.tab_gen_load
        t.columnconfigure(0, weight=1)
        t.columnconfigure(1, weight=3)
        t.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(t, text="Input Parameters – Q13 / Q14")
        left.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        left.columnconfigure(1, weight=1)

        params = [
            ("Excitation E\u2080 (V/phase)", self.gen_e0, 500.0, 6000.0),
            ("Sync. reactance Xs (\u03a9)", self.gen_xs, 0.5, 30.0),
            ("Resistive load R (\u03a9) [Q13]", self.gen_r, 0.1, 50.0),
        ]
        for r, (name, var, lo, hi) in enumerate(params):
            ttk.Label(left, text=name).grid(row=r, column=0, sticky="w", padx=4, pady=3)
            value_lbl = ttk.Label(left, width=10, text=f"{var.get():.2f}")
            value_lbl.grid(row=r, column=2, sticky="e")
            ttk.Scale(
                left,
                from_=lo,
                to=hi,
                variable=var,
                orient="horizontal",
                length=200,
                command=lambda _v, vv=var, ll=value_lbl: (
                    ll.config(text=f"{vv.get():.2f}"),
                    self.update_gen_load_plot(),
                ),
            ).grid(row=r, column=1, padx=4)

        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=len(params), column=0, columnspan=3, pady=6)
        ttk.Button(btn_frame, text="Start", command=self.update_gen_load_plot).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Stop",
                   command=lambda: self.status_var.set("Generator analysis paused.")).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Reset", command=self._reset_gen_load).pack(side="left", padx=4)

        self.gen_text = tk.Text(left, wrap="word", font=("Consolas", 9))
        self.gen_text.grid(row=len(params) + 1, column=0, columnspan=3,
                           sticky="nsew", padx=4, pady=4)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.gen_text.yview)
        sb.grid(row=len(params) + 1, column=3, sticky="ns")
        self.gen_text.configure(yscrollcommand=sb.set)
        left.rowconfigure(len(params) + 1, weight=1)

        right = ttk.LabelFrame(t, text="Visualization – Q13 / Q14")
        right.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.gen_fig = Figure(figsize=(10, 8), dpi=100)
        self.gen_canvas = FigureCanvasTkAgg(self.gen_fig, master=right)
        self.gen_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def update_gen_load_plot(self) -> None:
        E0 = self.gen_e0.get()
        Xs = max(self.gen_xs.get(), EPS)
        R_q13 = max(self.gen_r.get(), EPS)

        # Q13: single operating point ----------------------------------------
        denom_q13 = np.sqrt(R_q13 ** 2 + Xs ** 2)
        I_q13 = E0 / denom_q13
        E_q13 = I_q13 * R_q13
        P_q13 = I_q13 ** 2 * R_q13
        jXsI_q13 = Xs * I_q13

        # Q14: table of fixed R values ----------------------------------------
        fixed_R_raw = [None, 24.0, 12.0, 6.0, 3.0, 0.0]   # None = open circuit
        R_labels = ["\u221e", "24", "12", "6", "3", "0"]
        I_tbl: list[float] = []
        E_tbl: list[float] = []
        P_tbl: list[float] = []
        for Rv in fixed_R_raw:
            if Rv is None:                          # open circuit
                I_tbl.append(0.0)
                E_tbl.append(E0)
                P_tbl.append(0.0)
            elif Rv == 0.0:                         # short circuit
                I_tbl.append(E0 / Xs)
                E_tbl.append(0.0)
                P_tbl.append(0.0)
            else:
                Ii = E0 / np.sqrt(Rv ** 2 + Xs ** 2)
                I_tbl.append(Ii)
                E_tbl.append(Ii * Rv)
                P_tbl.append(Ii ** 2 * Rv)

        # Maximum power (R_opt = Xs) -----------------------------------------
        R_opt = Xs
        P_max = E0 ** 2 / (2.0 * Xs)

        # Continuous parametric curves (R: EPS → 150 Ω) ----------------------
        R_c = np.linspace(EPS, 150.0, 1000)
        I_c = E0 / np.sqrt(R_c ** 2 + Xs ** 2)
        E_c = I_c * R_c
        P_c = (I_c ** 2) * R_c

        # ── build figure ────────────────────────────────────────────────────
        self.gen_fig.clear()
        ax1 = self.gen_fig.add_subplot(221)
        ax2 = self.gen_fig.add_subplot(222)
        ax3 = self.gen_fig.add_subplot(223)
        ax4 = self.gen_fig.add_subplot(224)

        # ── ax1: Phasor diagram (Q13) ──
        pad = E0 * 0.13
        ax1.annotate("", xy=(E_q13, 0), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2.5))
        ax1.annotate("", xy=(E_q13, jXsI_q13), xytext=(E_q13, 0),
                     arrowprops=dict(arrowstyle="->", color="tab:orange", lw=2.5))
        ax1.annotate("", xy=(E_q13, jXsI_q13), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.5))
        ax1.annotate("", xy=(min(I_q13 * 0.28, E_q13 * 0.35), 0), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="tab:gray", lw=1.5))
        ax1.text(E_q13 / 2.0, -pad * 0.55,
                 f"E = {E_q13:.1f} V", ha="center", color="tab:blue",
                 fontsize=9, fontweight="bold")
        ax1.text(E_q13 + pad * 0.15, jXsI_q13 / 2.0,
                 f"jXsI = {jXsI_q13:.1f} V", color="tab:orange", fontsize=9)
        ax1.text(E_q13 * 0.25, jXsI_q13 * 0.72,
                 f"E\u2080 = {E0:.1f} V", color="tab:red", fontsize=9, fontweight="bold")
        ax1.text(min(I_q13 * 0.14, E_q13 * 0.18), -pad * 0.3,
                 "I (ref)", color="tab:gray", fontsize=8)
        ax1.set_xlim(-pad * 0.4, E_q13 + pad * 1.6)
        ax1.set_ylim(-pad * 0.8, jXsI_q13 + pad * 0.8)
        ax1.axhline(0, color="k", lw=0.5)
        ax1.axvline(0, color="k", lw=0.5)
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_title(
            f"Q13: Phasor Diagram  (R = {R_q13:.1f} \u03a9)\n"
            f"I = {I_q13:.2f} A  |  E = {E_q13:.2f} V  |  E\u2080 = {E0:.1f} V",
            fontsize=9,
        )
        ax1.set_xlabel("Real axis (V)")
        ax1.set_ylabel("Imaginary axis (V)")
        ax1.grid(alpha=0.3)

        # ── ax2: E vs I (Q14a) ──
        ax2.plot(I_c, E_c, lw=2, color="tab:blue", label="E vs I  (variable R)")
        ax2.scatter(I_tbl[1:-1], E_tbl[1:-1],
                    color="tab:red", zorder=5, s=60, label="Fixed R values")
        ax2.scatter([0.0], [E0], marker="^", s=90, color="tab:green", zorder=6,
                    label="R = \u221e  (open circuit)")
        ax2.scatter([E0 / Xs], [0.0], marker="s", s=80, color="tab:purple", zorder=6,
                    label="R = 0  (short circuit)")
        ax2.scatter([I_q13], [E_q13], marker="*", s=200, color="tab:red", zorder=7,
                    label=f"Q13: R = {R_q13:.1f} \u03a9")
        for Ii, Ei, rl in zip(I_tbl[1:-1], E_tbl[1:-1], R_labels[1:-1]):
            ax2.annotate(f"  {rl}\u03a9", (Ii, Ei), fontsize=7, color="tab:gray")
        ax2.set_title("Q14a: Terminal Voltage E  vs  Load Current I")
        ax2.set_xlabel("Load Current  I (A)")
        ax2.set_ylabel("Terminal Voltage  E (V)")
        ax2.set_xlim(-E0 / Xs * 0.04, E0 / Xs * 1.08)
        ax2.set_ylim(-E0 * 0.04, E0 * 1.08)
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)

        # ── ax3: Power per phase bar chart (Q14b) ──
        p_kw = [p / 1000.0 for p in P_tbl]
        ax3.bar(R_labels, p_kw, color="tab:steelblue")
        p_max_kw = P_max / 1000.0
        for j, pv in enumerate(p_kw):
            ax3.text(j, pv + p_max_kw * 0.015,
                     f"{pv:.1f}", ha="center", fontsize=8)
        ax3.axhline(p_max_kw, ls="--", color="tab:red",
                    label=f"P_max = {p_max_kw:.1f} kW @ R = Xs = {Xs:.1f} \u03a9")
        ax3.set_title("Q14b: Active Power P per Phase")
        ax3.set_xlabel("Resistance  R (\u03a9)")
        ax3.set_ylabel("Power (kW)")
        ax3.set_ylim(0, p_max_kw * 1.18)
        ax3.grid(alpha=0.25, axis="y")
        ax3.legend(fontsize=8)

        # ── ax4: E vs P (Q14c) ──
        ax4.plot(P_c / 1000.0, E_c, lw=2, color="tab:green", label="E vs P  (variable R)")
        ax4.scatter([p / 1000.0 for p in P_tbl[1:-1]], E_tbl[1:-1],
                    color="tab:red", zorder=5, s=60)
        ax4.axvline(p_max_kw, ls="--", color="tab:purple",
                    label=f"P_max = {p_max_kw:.1f} kW\n@ R = Xs = {Xs:.1f} \u03a9")
        for Pv, Ei, rl in zip(P_tbl[1:-1], E_tbl[1:-1], R_labels[1:-1]):
            ax4.annotate(f"  {rl}\u03a9", (Pv / 1000.0, Ei), fontsize=7, color="tab:gray")
        ax4.set_title("Q14c: Terminal Voltage E  vs  Power P")
        ax4.set_xlabel("Power per Phase  P (kW)")
        ax4.set_ylabel("Terminal Voltage  E (V)")
        ax4.grid(alpha=0.3)
        ax4.legend(fontsize=8)

        self.gen_fig.tight_layout()
        self.gen_canvas.draw_idle()

        self._update_gen_text(
            E0, Xs, R_q13, I_q13, E_q13, P_q13, jXsI_q13,
            R_labels, I_tbl, E_tbl, P_tbl, R_opt, P_max,
        )
        self.status_var.set(
            f"Q13: E = {E_q13:.1f} V, I = {I_q13:.1f} A, P = {P_q13/1000.0:.2f} kW  |  "
            f"P_max = {P_max/1000.0:.1f} kW @ R = Xs = {Xs:.1f} \u03a9"
        )

    def _update_gen_text(
        self,
        E0: float,
        Xs: float,
        R_q13: float,
        I_q13: float,
        E_q13: float,
        P_q13: float,
        jXsI_q13: float,
        R_labels: list,
        I_tbl: list,
        E_tbl: list,
        P_tbl: list,
        R_opt: float,
        P_max: float,
    ) -> None:
        lines = [
            "=" * 60 + "\n",
            " QUESTION 13 – 3-Phase Generator with Resistive Load\n",
            "=" * 60 + "\n",
            "\n",
            "GIVEN:\n",
            f"  Synchronous reactance  Xs = {Xs:.4f} \u03a9\n",
            f"  Excitation voltage     E\u2080 = {E0:.2f} V (line-to-neutral)\n",
            f"  Resistive load         R  = {R_q13:.4f} \u03a9\n",
            "\n",
            "CIRCUIT EQUATION (phasor):\n",
            "  E\u2080 = E + jXs\u00b7I          (Kirchhoff's voltage law)\n",
            "  Resistive load \u21d2 E = I\u00b7R  (E in phase with I)\n",
            "  \u2234  E\u2080 = I\u00b7R + jXs\u00b7I = I\u00b7(R + jXs)\n",
            "\n",
            "SOLUTION:\n",
            "  |E\u2080|\u00b2 = |I|\u00b2 \u00b7 (R\u00b2 + Xs\u00b2)\n",
            "  |I|  = |E\u2080| / \u221a(R\u00b2 + Xs\u00b2)\n",
            f"       = {E0:.2f} / \u221a({R_q13:.4f}\u00b2 + {Xs:.4f}\u00b2)\n",
            f"       = {E0:.2f} / \u221a({R_q13**2:.4f} + {Xs**2:.4f})\n",
            f"       = {E0:.2f} / {np.sqrt(R_q13**2 + Xs**2):.6f}\n",
            f"       = {I_q13:.6f} A  \u2248  {I_q13:.2f} A\n",
            "\n",
            f"  |E|  = I\u00b7R = {I_q13:.6f} \u00d7 {R_q13:.4f}\n",
            f"       = {E_q13:.6f} V  \u2248  {E_q13:.2f} V  (line-to-neutral)\n",
            f"  Line-to-line voltage = E\u00b7\u221a3 = {E_q13 * np.sqrt(3.0):.2f} V\n",
            "\n",
            "PHASOR DIAGRAM (top-left plot):\n",
            "  \u2022 Reference: current I along +real axis\n",
            f"  \u2022 E  = {E_q13:.2f} V  along +real (resistive \u21d2 in phase with I)\n",
            f"  \u2022 jXs\u00b7I = {jXsI_q13:.2f} V  along +imag (leads I by 90\u00b0)\n",
            f"  \u2022 E\u2080 = \u221a(E\u00b2 + (Xs\u00b7I)\u00b2) = \u221a({E_q13**2:.2f} + {jXsI_q13**2:.2f})\n",
            f"       = {np.sqrt(E_q13**2 + jXsI_q13**2):.4f} V  \u2713 checks with E\u2080 = {E0:.2f} V\n",
            "\n",
            "ACTIVE POWER per phase:\n",
            f"  P = I\u00b2\u00b7R = {I_q13:.6f}\u00b2 \u00d7 {R_q13:.4f}\n",
            f"    = {P_q13:.4f} W  =  {P_q13 / 1000.0:.6f} kW\n",
            "\n",
            "=" * 60 + "\n",
            " QUESTION 14 – E vs I and E vs P Curves\n",
            "=" * 60 + "\n",
            "\n",
            "a) E vs I for various resistive loads:\n",
            f"   {'R (\u03a9)':<12}{'I (A)':<16}{'E (V)':<16}\n",
            "   " + "\u2500" * 44 + "\n",
        ]
        for rl, Ii, Ei in zip(R_labels, I_tbl, E_tbl):
            lines.append(f"   {rl:<12}{Ii:<16.4f}{Ei:<16.4f}\n")

        lines += [
            "\n",
            "b) Active power P per phase:\n",
            "   Formula: P = I\u00b2\u00b7R = E\u2080\u00b2\u00b7R / (R\u00b2 + Xs\u00b2)\n\n",
            f"   {'R (\u03a9)':<12}{'P (W)':<16}{'P (kW)':<16}\n",
            "   " + "\u2500" * 44 + "\n",
        ]
        for rl, Pv in zip(R_labels, P_tbl):
            lines.append(f"   {rl:<12}{Pv:<16.4f}{Pv / 1000.0:<16.6f}\n")

        lines += [
            "\n",
            "c) Maximum power condition:\n",
            "   P(R) = E\u2080\u00b2\u00b7R / (R\u00b2 + Xs\u00b2)\n",
            "   dP/dR = E\u2080\u00b2\u00b7(Xs\u00b2 \u2212 R\u00b2) / (R\u00b2 + Xs\u00b2)\u00b2  =  0\n",
            f"   \u2192  R_opt = Xs = {R_opt:.4f} \u03a9\n",
            "\n",
            "   P_max = E\u2080\u00b2 / (2\u00b7Xs)\n",
            f"         = {E0:.2f}\u00b2 / (2 \u00d7 {Xs:.4f})\n",
            f"         = {E0 ** 2:.4f} / {2.0 * Xs:.4f}\n",
            f"         = {P_max:.4f} W  =  {P_max / 1000.0:.6f} kW  (per phase)\n",
            f"   3-phase P_max = 3 \u00d7 {P_max / 1000.0:.6f}\n",
            f"               = {3.0 * P_max / 1000.0:.6f} kW\n",
            "\n",
            "PHYSICAL INSIGHT:\n",
            "  \u2022 Maximum power transfer when R = |jXs| = Xs\n",
            "    (matched impedance for purely reactive source Z).\n",
            "  \u2022 E vs I traces a QUARTER-CIRCLE (proof):\n",
            "    E = I\u00b7R,  I\u00b7Xs = I\u00b7Xs,  and I = E\u2080/\u221a(R\u00b2+Xs\u00b2)\n",
            "    \u21d2 E\u00b2 + (I\u00b7Xs)\u00b2 = (I\u00b7R)\u00b2 + (I\u00b7Xs)\u00b2 = I\u00b2(R\u00b2+Xs\u00b2) = E\u2080\u00b2\n",
            "    \u21d2 E\u00b2 + (I\u00b7Xs)\u00b2 = E\u2080\u00b2   (circle of radius E\u2080).\n",
            "  \u2022 As R\u2192\u221e:  I\u21920, E\u2192E\u2080  (no-load voltage = excitation).\n",
            "  \u2022 Short-circuit (R = 0):  all voltage across Xs,\n",
            f"    I_sc = E\u2080/Xs = {E0:.2f}/{Xs:.4f} = {E0 / Xs:.4f} A.\n",
        ]
        self.gen_text.delete("1.0", "end")
        self.gen_text.insert("1.0", "".join(lines))

    def _reset_gen_load(self) -> None:
        self.gen_e0.set(3000.0)
        self.gen_xs.set(6.0)
        self.gen_r.set(8.0)
        self.update_gen_load_plot()

    def _reset_defaults(self) -> None:
        self.mva.set(250.0)
        self.kv.set(25.0)
        self.freq.set(60.0)
        self.xd.set(1.6)
        self.xd_prime.set(0.23)
        self.p_pu.set(1.0)
        self.t_transient.set(0.18)
        self.dc_tau.set(0.06)
        self.update_all()


if __name__ == "__main__":
    app = AdvancedPowerSystemApp()
    app.mainloop()
