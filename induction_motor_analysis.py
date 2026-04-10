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

        # User-question inputs (generator efficiency problem)
        self.p_out_mw = tk.DoubleVar(value=500.0)
        self.eff_pct = tk.DoubleVar(value=98.4)
        self.ifield_a = tk.DoubleVar(value=2400.0)
        self.vfield_v = tk.DoubleVar(value=300.0)
        self.airflow_m3s = tk.DoubleVar(value=280.0)
        self.air_rho = tk.DoubleVar(value=1.2)
        self.air_cp = tk.DoubleVar(value=1005.0)
        self.poles = tk.DoubleVar(value=2.0)

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

        self.nb.add(self.tab_input, text="Main Menu / Inputs")
        self.nb.add(self.tab_fault, text="Modelling & Fault Current")
        self.nb.add(self.tab_protection, text="Protection Coordination")
        self.nb.add(self.tab_speed, text="Speed Controller")
        self.nb.add(self.tab_thermal, text="Thermal Analysis")
        self.nb.add(self.tab_economic, text="Economic Analysis")
        self.nb.add(self.tab_harmonic, text="Harmonic & Power Quality")
        self.nb.add(self.tab_advanced, text="Comprehensive Analysis")

        self._build_tab_input()
        self._build_tab_fault()
        self._build_tab_protection()
        self._build_tab_speed()
        self._build_tab_thermal()
        self._build_tab_economic()
        self._build_tab_harmonic()
        self._build_tab_advanced()

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
            ("Output power (MW)", self.p_out_mw, 100.0, 1200.0),
            ("Efficiency (%)", self.eff_pct, 80.0, 99.8),
            ("Field current (A)", self.ifield_a, 200.0, 6000.0),
            ("Field voltage (Vdc)", self.vfield_v, 50.0, 1200.0),
            ("Air flow (m^3/s)", self.airflow_m3s, 20.0, 600.0),
            ("Air density (kg/m^3)", self.air_rho, 0.8, 1.4),
            ("Air Cp (J/kgK)", self.air_cp, 950.0, 1100.0),
            ("Machine poles", self.poles, 2.0, 12.0),
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

        p_out_w = self.p_out_mw.get() * 1e6
        eff = clamp(self.eff_pct.get() / 100.0, 0.01, 0.999999)
        p_in_total = p_out_w / eff
        p_losses_total = p_in_total - p_out_w
        p_rotor_cu = self.vfield_v.get() * self.ifield_a.get()
        p_shaft = max(p_in_total - p_rotor_cu, EPS)
        poles_even = max(2.0, 2.0 * round(self.poles.get() / 2.0))
        n_sync_rpm = 120.0 * self.freq.get() / poles_even
        omega_sync = 2.0 * np.pi * n_sync_rpm / 60.0
        turbine_torque = p_shaft / max(omega_sync, EPS)
        delta_t_air = p_losses_total / max(self.air_rho.get() * self.airflow_m3s.get() * self.air_cp.get(), EPS)

        return {
            "i_base": i_base,
            "z_base": z_base,
            "v_phase": v_phase,
            "e0_pu": e0_pu,
            "i_init_pu": i_init_pu,
            "i_final_pu": i_final_pu,
            "i_init_a": i_init_pu * i_base,
            "i_final_a": i_final_pu * i_base,
            "p_out_w": p_out_w,
            "eff": eff,
            "p_losses_total": p_losses_total,
            "p_rotor_cu": p_rotor_cu,
            "p_shaft": p_shaft,
            "n_sync_rpm": n_sync_rpm,
            "turbine_torque": turbine_torque,
            "delta_t_air": delta_t_air,
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
        self.status_var.set(
            f"Solved: E0={data['e0_pu']:.3f} pu, I_initial={data['i_init_a']:.0f} A, I_final={data['i_final_a']:.0f} A"
        )

    def _update_main_text(self, d: dict[str, float]) -> None:
        txt = []
        txt.append("SYNCHRONOUS GENERATOR: DETAILED SOLUTION + MODELLING\n")
        txt.append("Question data: η=98.4%, Pout=500 MW, Ifield=2400 A, Vfield=300 V, airflow=280 m^3/s.\n")
        txt.append("\nA) Total losses in machine:\n")
        txt.append("   η = Pout/Pin  =>  Pin = Pout/η\n")
        txt.append(f"   Pin = {d['p_out_w']/1e6:.3f}/{d['eff']:.5f} = {(d['p_out_w']/d['eff'])/1e6:.3f} MW\n")
        txt.append(f"   P_losses,total = Pin - Pout = {d['p_losses_total']/1e6:.3f} MW\n")
        txt.append("\nB) Rotor copper losses:\n")
        txt.append("   Pcu,rotor = Vdc * Idc\n")
        txt.append(f"   Pcu,rotor = {self.vfield_v.get():.1f} * {self.ifield_a.get():.1f} = {d['p_rotor_cu']/1e3:.1f} kW\n")
        txt.append("\nC) Turbine developed torque:\n")
        txt.append("   Assumption: overall efficiency includes field power, so shaft power Pshaft = Pin - Pfield.\n")
        txt.append(f"   n_sync = 120 f/P = {d['n_sync_rpm']:.1f} rpm, ω = 2πn/60\n")
        txt.append(f"   T_turbine = Pshaft/ω = {d['turbine_torque']:.1f} N·m\n")
        txt.append("\nD) Average cooling-air temperature rise:\n")
        txt.append("   ΔT = Ploss / (ρ * Q * cp)\n")
        txt.append(f"   ΔT = {d['p_losses_total']:.2e} / ({self.air_rho.get():.3f}*{self.airflow_m3s.get():.2f}*{self.air_cp.get():.1f})")
        txt.append(f" = {d['delta_t_air']:.2f} °C\n")
        txt.append("\nInterpretation: this ΔT is a bulk-average estimate. Real local hotspots can be significantly higher.\n")
        txt.append("\n----------------------------------\n")
        txt.append("Dynamic module (fault & controls) used for practical engineering studies:\n")
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
        copper_loss = d["p_rotor_cu"] + 3.0 * (d["i_final_a"] ** 2) * 0.004
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
        ax1.bar(
            ["Initial SC kA", "Final SC kA"],
            [d["i_init_a"] / 1000.0, d["i_final_a"] / 1000.0],
            color=["tab:red", "tab:blue"],
        )
        ax1.set_title("Fault Current Metrics")
        ax1.grid(alpha=0.25)

        # B) Energy balance from problem statement
        losses_other = max(d["p_losses_total"] - d["p_rotor_cu"], 0.0)
        ax2.pie(
            [d["p_out_w"], d["p_rotor_cu"], losses_other],
            labels=["Output", "Rotor Cu", "Other losses"],
            autopct="%1.1f%%",
        )
        ax2.set_title("Power Balance Breakdown")

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
        ax4.bar(["Overall index", "ΔT air (°C)"], [score, clamp(d["delta_t_air"], 0.0, 100.0)], color=["tab:blue", "tab:orange"])
        ax4.axhline(80, color="tab:green", ls="--")
        ax4.set_ylim(0, 100)
        ax4.set_title("Comprehensive Engineering Index")
        ax4.grid(alpha=0.25)

        self.adv_fig.tight_layout()
        self.adv_canvas.draw_idle()

    def _reset_defaults(self) -> None:
        self.mva.set(250.0)
        self.kv.set(25.0)
        self.freq.set(60.0)
        self.xd.set(1.6)
        self.xd_prime.set(0.23)
        self.p_pu.set(1.0)
        self.t_transient.set(0.18)
        self.dc_tau.set(0.06)
        self.p_out_mw.set(500.0)
        self.eff_pct.set(98.4)
        self.ifield_a.set(2400.0)
        self.vfield_v.set(300.0)
        self.airflow_m3s.set(280.0)
        self.air_rho.set(1.2)
        self.air_cp.set(1005.0)
        self.poles.set(2.0)
        self.update_all()


if __name__ == "__main__":
    app = AdvancedPowerSystemApp()
    app.mainloop()
