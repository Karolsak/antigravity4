"""
Wound-Rotor Induction Motor Analysis Application
400V, 1450rpm, 50Hz, 4-pole wound-rotor induction motor
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq


# ─── Motor Physics ────────────────────────────────────────────────────────────

def equivalent_circuit(V1, f, R1, R2p, X1, X2p, Xm, s, Re=0.0):
    """Compute per-phase quantities using exact equivalent circuit."""
    if abs(s) < 1e-9:
        s = 1e-9
    omega_s = 2 * np.pi * f
    jX1 = 1j * X1
    jX2p = 1j * X2p
    jXm = 1j * Xm
    R2s = (R2p + Re) / s
    Z2 = R2s + jX2p
    Zm = jXm
    Zin = jX1 + R1 + (Zm * Z2) / (Zm + Z2)
    I1 = V1 / Zin
    # Voltage across Xm branch
    Vm = V1 - I1 * (R1 + jX1)
    I2p = Vm / Z2
    return I1, I2p, Vm, Zin

def compute_torque(V1, f, R1, R2p, X1, X2p, Xm, s, poles=4, Re=0.0):
    """Torque from air-gap power."""
    if abs(s) < 1e-9:
        s = 1e-9
    omega_s = 2 * np.pi * f / (poles / 2)
    I1, I2p, Vm, Zin = equivalent_circuit(V1, f, R1, R2p, X1, X2p, Xm, s, Re)
    Pag = 3 * abs(I2p)**2 * (R2p + Re) / s
    Te = Pag / omega_s
    return Te, Pag

def motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm, s, poles=4, Prot=1500, Re=0.0):
    """Full parameter set at a given slip."""
    omega_s = 2 * np.pi * f / (poles / 2)
    Te, Pag = compute_torque(V1, f, R1, R2p, X1, X2p, Xm, s, poles, Re)
    I1, I2p, Vm, Zin = equivalent_circuit(V1, f, R1, R2p, X1, X2p, Xm, s, Re)
    omega_r = omega_s * (1 - s)
    Pmech = Pag * (1 - s)
    Pout = Pmech - Prot
    Pin = 3 * V1 * abs(I1) * np.cos(np.angle(I1))
    eta_internal = Pmech / Pag if Pag > 0 else 0
    eta_overall = Pout / Pin if Pin > 0 else 0
    pf = np.cos(np.angle(I1))
    speed_rpm = omega_r * 60 / (2 * np.pi)
    Tnet = Pout / omega_r if omega_r > 0 else 0
    return {
        'I1': I1, 'I2p': I2p, 'Te': Te, 'Tnet': Tnet,
        'Pag': Pag, 'Pmech': Pmech, 'Pout': Pout, 'Pin': Pin,
        'eta_int': eta_internal, 'eta_overall': eta_overall,
        'pf': pf, 'speed_rpm': speed_rpm, 'omega_r': omega_r,
        'Vm': Vm, 'Zin': Zin
    }

def slip_max_torque(R1, R2p, X1, X2p, Xm, Re=0.0):
    """Analytical slip at maximum torque (Thevenin equivalent)."""
    Zth_num = 1j * Xm * (R1 + 1j * X1)
    Zth_den = R1 + 1j * (X1 + Xm)
    Zth = Zth_num / Zth_den
    Rth = Zth.real
    Xth = Zth.imag
    s_mt = (R2p + Re) / np.sqrt(Rth**2 + (Xth + X2p)**2)
    return s_mt


# ─── Main Application ─────────────────────────────────────────────────────────

class MotorAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wound-Rotor Induction Motor Analysis")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        self.configure(bg='#f0f0f0')

        # Motor base parameters
        self.VL = tk.DoubleVar(value=400.0)
        self.freq = tk.DoubleVar(value=50.0)
        self.R1 = tk.DoubleVar(value=0.3)
        self.R2p = tk.DoubleVar(value=0.25)
        self.X1 = tk.DoubleVar(value=0.6)
        self.X2p = tk.DoubleVar(value=0.6)
        self.Xm = tk.DoubleVar(value=35.0)
        self.Prot = tk.DoubleVar(value=1500.0)
        self.poles = 4
        self.Re_ext = tk.DoubleVar(value=0.0)

        # Simulation state
        self.sim_running = False
        self.sim_anim = None
        self.speed_ctrl_running = False
        self.speed_ctrl_anim = None

        self._build_menu()
        self._build_notebook()
        self._build_statusbar()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.status_var.set("Ready. Motor: 400V, 50Hz, 4-pole wound-rotor induction motor.")

    # ── Menu ────────────────────────────────────────────────────────────────
    def _build_menu(self):
        mb = tk.Menu(self)
        self.config(menu=mb)

        file_m = tk.Menu(mb, tearoff=0)
        file_m.add_command(label="Exit", command=self._on_close)
        mb.add_cascade(label="File", menu=file_m)

        view_m = tk.Menu(mb, tearoff=0)
        tabs = ["Input Parameters", "Torque-Speed", "Simulation",
                "Fault Current", "Protection", "Speed Controller",
                "Thermal", "Economic", "Harmonics", "Comprehensive"]
        for i, t in enumerate(tabs):
            view_m.add_command(label=t, command=lambda idx=i: self.nb.select(idx))
        mb.add_cascade(label="View", menu=view_m)

        help_m = tk.Menu(mb, tearoff=0)
        help_m.add_command(label="About", command=self._about)
        mb.add_cascade(label="Help", menu=help_m)

    def _about(self):
        messagebox.showinfo("About", "Wound-Rotor Induction Motor Analysis\n"
                            "400V, 1450rpm, 50Hz, 4-pole\n"
                            "R1=0.3Ω  R2'=0.25Ω  X1=X2'=0.6Ω  Xm=35Ω\n"
                            "Prot=1500W\n\nBuilt with Python/Tkinter/Matplotlib")

    def _build_statusbar(self):
        self.status_var = tk.StringVar()
        sb = ttk.Label(self, textvariable=self.status_var, relief='sunken', anchor='w')
        sb.grid(row=2, column=0, sticky='ew', padx=2, pady=2)

    # ── Notebook ─────────────────────────────────────────────────────────────
    def _build_notebook(self):
        self.nb = ttk.Notebook(self)
        self.nb.grid(row=1, column=0, sticky='nsew', padx=4, pady=4)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self._build_tab_input()
        self._build_tab_torque_speed()
        self._build_tab_simulation()
        self._build_tab_fault()
        self._build_tab_protection()
        self._build_tab_speed_ctrl()
        self._build_tab_thermal()
        self._build_tab_economic()
        self._build_tab_harmonics()
        self._build_tab_comprehensive()

    def _get_V1(self):
        return self.VL.get() / np.sqrt(3)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 – Input Parameters
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_input(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" Input Parameters ")
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(1, weight=1)

        # ── Sliders panel ──
        sf = ttk.LabelFrame(tab, text="Motor Parameters")
        sf.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)

        params = [
            ("R1 (Ω)", self.R1, 0.01, 2.0),
            ("R2' (Ω)", self.R2p, 0.01, 2.0),
            ("X1 (Ω)", self.X1, 0.1, 5.0),
            ("X2' (Ω)", self.X2p, 0.1, 5.0),
            ("Xm (Ω)", self.Xm, 5.0, 100.0),
            ("Prot (W)", self.Prot, 0, 5000),
            ("VL (V)", self.VL, 100, 690),
            ("Freq (Hz)", self.freq, 10, 60),
        ]
        self._inp_labels = {}
        for r, (name, var, lo, hi) in enumerate(params):
            ttk.Label(sf, text=name, width=10).grid(row=r, column=0, sticky='e', padx=4, pady=3)
            lbl = ttk.Label(sf, text=f"{var.get():.3f}", width=8)
            lbl.grid(row=r, column=2, sticky='w', padx=4)
            self._inp_labels[name] = lbl
            sl = ttk.Scale(sf, from_=lo, to=hi, variable=var, orient='horizontal', length=220,
                           command=lambda v, n=name, l=lbl, vr=var: (
                               l.config(text=f"{vr.get():.3f}"), self._calc_and_display()))
            sl.grid(row=r, column=1, sticky='ew', padx=4, pady=3)

        ttk.Button(sf, text="Calculate", command=self._calc_and_display).grid(
            row=len(params), column=0, columnspan=3, pady=8)

        # ── Results text ──
        rf = ttk.LabelFrame(tab, text="Results & Explanation")
        rf.grid(row=0, column=1, rowspan=2, sticky='nsew', padx=6, pady=6)
        rf.rowconfigure(0, weight=1)
        rf.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        self.result_text = tk.Text(rf, wrap='word', font=('Courier', 10), bg='#1e1e1e', fg='#d4d4d4',
                                    insertbackground='white', relief='flat', padx=8, pady=8)
        scr = ttk.Scrollbar(rf, orient='vertical', command=self.result_text.yview)
        self.result_text['yscrollcommand'] = scr.set
        self.result_text.grid(row=0, column=0, sticky='nsew')
        scr.grid(row=0, column=1, sticky='ns')

        self._calc_and_display()

    def _calc_and_display(self):
        try:
            V1 = self._get_V1()
            f = self.freq.get()
            R1 = self.R1.get(); R2p = self.R2p.get()
            X1 = self.X1.get(); X2p = self.X2p.get()
            Xm = self.Xm.get(); Prot = self.Prot.get()
            poles = self.poles
            ns = 120 * f / poles  # sync speed rpm
            omega_s = 2 * np.pi * f / (poles / 2)

            # (a) Starting (s=1)
            pa = motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm, s=1.0, poles=poles, Prot=Prot)
            I_start = abs(pa['I1'])
            T_start = pa['Te']

            # (b) Full-load s=0.0333
            s_fl = 0.0333
            pb = motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm, s=s_fl, poles=poles, Prot=Prot)

            # (c) Max torque
            s_mt = slip_max_torque(R1, R2p, X1, X2p, Xm)
            pc = motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm, s=s_mt, poles=poles, Prot=Prot)
            T_max = pc['Te']

            lines = []
            lines.append("=" * 60)
            lines.append(" WOUND-ROTOR INDUCTION MOTOR ANALYSIS")
            lines.append("=" * 60)
            lines.append(f" Supply: {self.VL.get():.1f} V (L-L), {f:.1f} Hz")
            lines.append(f" V1(phase) = {V1:.2f} V")
            lines.append(f" Sync speed = {ns:.0f} rpm  (ωs = {omega_s:.3f} rad/s)")
            lines.append(f" R1={R1}Ω  R2'={R2p}Ω  X1={X1}Ω  X2'={X2p}Ω  Xm={Xm}Ω")
            lines.append(f" Prot = {Prot:.0f} W")
            lines.append("")
            lines.append("─" * 60)
            lines.append("(a) STARTING CONDITIONS  (s = 1.0, direct-on-line)")
            lines.append("─" * 60)
            lines.append(f"  Starting current  I_start = {I_start:.2f} A (per phase)")
            lines.append(f"  3-phase line current = {I_start:.2f} A")
            lines.append(f"  Starting torque   T_start = {T_start:.2f} N·m")
            lines.append(f"  Air-gap power     Pag     = {pa['Pag']:.1f} W")
            lines.append(f"  Power factor      pf      = {pa['pf']:.3f}")
            lines.append("")
            lines.append("─" * 60)
            lines.append(f"(b) FULL-LOAD CONDITIONS  (s = {s_fl})")
            lines.append("─" * 60)
            lines.append(f"  Speed         = {pb['speed_rpm']:.1f} rpm")
            lines.append(f"  Stator current I1 = {abs(pb['I1']):.3f} A ∠{np.degrees(np.angle(pb['I1'])):.1f}°")
            lines.append(f"  Power factor   pf = {pb['pf']:.4f} (lagging)")
            lines.append(f"  Air-gap power  Pag   = {pb['Pag']:.1f} W")
            lines.append(f"  Mech power     Pmech = {pb['Pmech']:.1f} W")
            lines.append(f"  Net output     Pout  = {pb['Pout']:.1f} W  ({pb['Pout']/1000:.3f} kW)")
            lines.append(f"  Input power    Pin   = {pb['Pin']:.1f} W")
            lines.append(f"  Net torque     Tnet  = {pb['Tnet']:.2f} N·m")
            lines.append(f"  Internal eff.  η_int = {pb['eta_int']*100:.2f} %")
            lines.append(f"  Overall eff.   η     = {pb['eta_overall']*100:.2f} %")
            lines.append("")
            lines.append("─" * 60)
            lines.append("(c) MAXIMUM TORQUE CONDITIONS")
            lines.append("─" * 60)
            lines.append(f"  Slip at max torque s_mt = {s_mt:.4f}")
            lines.append(f"  Speed at max torque  = {pc['speed_rpm']:.1f} rpm")
            lines.append(f"  Maximum torque T_max = {T_max:.2f} N·m")
            lines.append(f"  Ratio T_max/T_FL  = {T_max/pb['Te']:.2f}")
            lines.append(f"  Ratio T_start/T_FL= {T_start/pb['Te']:.2f}")
            lines.append("")
            lines.append("─" * 60)
            lines.append(" EQUIVALENT CIRCUIT FORMULAE")
            lines.append("─" * 60)
            lines.append("  Zin = jX1 + R1 + Zm‖Z2")
            lines.append("  Z2  = R2'/s + jX2'")
            lines.append("  Zm  = jXm")
            lines.append("  I1  = V1/Zin")
            lines.append("  I2' = Vm/Z2  where Vm = V1 - I1(R1+jX1)")
            lines.append("  Pag = 3|I2'|²·R2'/s")
            lines.append("  Te  = Pag/ωs")
            lines.append("  η   = Pout/Pin")
            lines.append("=" * 60)

            self.result_text.config(state='normal')
            self.result_text.delete('1.0', 'end')
            self.result_text.insert('end', "\n".join(lines))
            self.result_text.config(state='disabled')
            self.status_var.set(f"Calculated: T_start={T_start:.1f}Nm  T_FL={pb['Te']:.1f}Nm  T_max={T_max:.1f}Nm  η={pb['eta_overall']*100:.1f}%")
        except Exception as e:
            self.status_var.set(f"Calculation error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 – Torque-Speed Characteristics
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_torque_speed(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" Torque-Speed ")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        # Canvas
        fig = Figure(figsize=(10, 6), dpi=96, facecolor='#f8f8f8')
        self._ts_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=tab)
        self._ts_canvas = canvas
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        # Controls
        ctrl = ttk.LabelFrame(tab, text="External Rotor Resistance (Wound-Rotor Control)")
        ctrl.grid(row=1, column=0, sticky='ew', padx=6, pady=4)
        ttk.Label(ctrl, text="Re (Ω):").grid(row=0, column=0, padx=6, pady=4)
        self._re_lbl = ttk.Label(ctrl, text="0.000 Ω", width=10)
        self._re_lbl.grid(row=0, column=2, padx=4)
        ttk.Scale(ctrl, from_=0, to=2, variable=self.Re_ext, orient='horizontal', length=300,
                  command=lambda v: (self._re_lbl.config(text=f"{self.Re_ext.get():.3f} Ω"),
                                     self._plot_torque_speed())).grid(row=0, column=1, padx=6)
        ttk.Button(ctrl, text="Refresh Plot", command=self._plot_torque_speed).grid(row=0, column=3, padx=10)

        info = ttk.Label(ctrl, text="Adding external resistance shifts the max-torque slip → lower speed at same torque (wound-rotor speed control).",
                         foreground='#555')
        info.grid(row=1, column=0, columnspan=4, sticky='w', padx=6, pady=2)

        tab.bind('<Configure>', lambda e: self._plot_torque_speed())
        self._plot_torque_speed()

    def _plot_torque_speed(self):
        try:
            fig = self._ts_fig
            fig.clf()
            ax = fig.add_subplot(111)
            V1 = self._get_V1(); f = self.freq.get()
            R1 = self.R1.get(); R2p = self.R2p.get()
            X1 = self.X1.get(); X2p = self.X2p.get()
            Xm = self.Xm.get(); Prot = self.Prot.get()
            poles = self.poles
            ns = 120 * f / poles
            Re = self.Re_ext.get()

            slips = np.linspace(0.001, 1.0, 500)
            speeds = ns * (1 - slips)

            for re, lbl, col in [(0.0, 'Re=0 (base)', '#2196F3'),
                                  (Re, f'Re={Re:.2f}Ω', '#F44336')]:
                torques = []
                for s in slips:
                    try:
                        Te, _ = compute_torque(V1, f, R1, R2p, X1, X2p, Xm, s, poles, re)
                        torques.append(Te)
                    except:
                        torques.append(0)
                ax.plot(speeds, torques, color=col, linewidth=2, label=lbl)

            # Mark key points (Re=0)
            s_fl = 0.0333
            pb = motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm, s_fl, poles, Prot, 0.0)
            ax.scatter([pb['speed_rpm']], [pb['Te']], color='green', s=80, zorder=5,
                       label=f"FL op ({pb['speed_rpm']:.0f}rpm, {pb['Te']:.1f}Nm)")

            pa = motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm, 1.0, poles, Prot, 0.0)
            ax.scatter([0], [pa['Te']], color='orange', s=80, zorder=5,
                       label=f"Start (0rpm, {pa['Te']:.1f}Nm)")

            s_mt = slip_max_torque(R1, R2p, X1, X2p, Xm, 0.0)
            pc = motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm, s_mt, poles, Prot, 0.0)
            ax.scatter([pc['speed_rpm']], [pc['Te']], color='red', s=80, zorder=5,
                       label=f"Max T ({pc['speed_rpm']:.0f}rpm, {pc['Te']:.1f}Nm)")

            ax.set_xlabel("Rotor Speed (rpm)", fontsize=11)
            ax.set_ylabel("Electromagnetic Torque (N·m)", fontsize=11)
            ax.set_title("Torque-Speed Characteristics (Wound-Rotor Induction Motor)", fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)
            ax.set_xlim(-50, ns + 50)
            fig.tight_layout()
            self._ts_canvas.draw()
        except Exception as e:
            self.status_var.set(f"Plot error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 – Interactive Simulation (Motor Dynamics)
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_simulation(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" Simulation ")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        fig = Figure(figsize=(10, 5), dpi=96, facecolor='#f8f8f8')
        self._sim_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=tab)
        self._sim_canvas = canvas
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        ctrl = ttk.LabelFrame(tab, text="Simulation Parameters")
        ctrl.grid(row=1, column=0, sticky='ew', padx=6, pady=4)

        self.J_var = tk.DoubleVar(value=0.5)
        self.TL_var = tk.DoubleVar(value=20.0)
        self.omega0_var = tk.DoubleVar(value=0.0)
        self.sim_duration = tk.DoubleVar(value=5.0)

        sliders = [
            ("Inertia J (kg·m²)", self.J_var, 0.05, 5.0),
            ("Load Torque TL (N·m)", self.TL_var, 0.0, 100.0),
            ("Init Speed (rad/s)", self.omega0_var, 0.0, 160.0),
            ("Duration (s)", self.sim_duration, 0.5, 20.0),
        ]
        for c, (name, var, lo, hi) in enumerate(sliders):
            ttk.Label(ctrl, text=name).grid(row=0, column=c * 2, padx=4)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var, orient='horizontal', length=150,
                      command=lambda v: None).grid(row=1, column=c * 2, padx=4, pady=2)

        btnf = ttk.Frame(ctrl)
        btnf.grid(row=2, column=0, columnspan=8, pady=4)
        ttk.Button(btnf, text="▶ Run Simulation", command=self._run_simulation).pack(side='left', padx=4)
        ttk.Button(btnf, text="⏹ Stop", command=self._stop_simulation).pack(side='left', padx=4)
        ttk.Button(btnf, text="↺ Reset", command=self._reset_simulation).pack(side='left', padx=4)

        info = ttk.Label(ctrl, text="Model: J·dω/dt = Te(ω) - TL  (electromechanical equation of motion)",
                         foreground='#555')
        info.grid(row=3, column=0, columnspan=8, sticky='w', padx=6)

    def _run_simulation(self):
        try:
            V1 = self._get_V1(); f = self.freq.get()
            R1 = self.R1.get(); R2p = self.R2p.get()
            X1 = self.X1.get(); X2p = self.X2p.get()
            Xm = self.Xm.get(); poles = self.poles
            J = self.J_var.get()
            TL = self.TL_var.get()
            omega0 = self.omega0_var.get()
            T = self.sim_duration.get()
            omega_s = 2 * np.pi * f / (poles / 2)

            def dyn(t, y):
                omega = y[0]
                s = max(1e-6, min(1.0, (omega_s - omega) / omega_s))
                try:
                    Te, _ = compute_torque(V1, f, R1, R2p, X1, X2p, Xm, s, poles)
                except:
                    Te = 0
                return [(Te - TL) / J]

            sol = solve_ivp(dyn, [0, T], [omega0], max_step=0.05, dense_output=True)
            t_arr = sol.t
            omega_arr = sol.y[0]
            speed_arr = omega_arr * 60 / (2 * np.pi)

            Te_arr = []
            for om in omega_arr:
                s = max(1e-6, min(1.0, (omega_s - om) / omega_s))
                try:
                    Te, _ = compute_torque(V1, f, R1, R2p, X1, X2p, Xm, s, poles)
                except:
                    Te = 0
                Te_arr.append(Te)

            fig = self._sim_fig
            fig.clf()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.plot(t_arr, speed_arr, 'b-', linewidth=2, label='Speed')
            ax1.axhline(120 * f / poles, color='r', ls='--', label=f'Sync {120*f/poles:.0f}rpm')
            ax1.set_ylabel("Speed (rpm)"); ax1.set_xlabel("Time (s)")
            ax1.set_title("Motor Speed Transient"); ax1.legend(); ax1.grid(True, alpha=0.4)

            ax2.plot(t_arr, Te_arr, 'g-', linewidth=2, label='Te')
            ax2.axhline(TL, color='r', ls='--', label=f'TL={TL}N·m')
            ax2.set_ylabel("Torque (N·m)"); ax2.set_xlabel("Time (s)")
            ax2.set_title("Electromagnetic Torque"); ax2.legend(); ax2.grid(True, alpha=0.4)

            fig.tight_layout()
            self._sim_canvas.draw()
            self.status_var.set(f"Simulation complete. Final speed: {speed_arr[-1]:.1f} rpm")
        except Exception as e:
            self.status_var.set(f"Simulation error: {e}")

    def _stop_simulation(self):
        self.sim_running = False
        self.status_var.set("Simulation stopped.")

    def _reset_simulation(self):
        self._sim_fig.clf()
        ax = self._sim_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Press 'Run Simulation' to start", ha='center', va='center',
                transform=ax.transAxes, fontsize=14, color='gray')
        self._sim_canvas.draw()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 – Fault Current Analysis
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_fault(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" Fault Current ")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        fig = Figure(figsize=(10, 6), dpi=96, facecolor='#f8f8f8')
        self._fault_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=tab)
        self._fault_canvas = canvas
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        ctrl = ttk.LabelFrame(tab, text="Fault Parameters")
        ctrl.grid(row=1, column=0, sticky='ew', padx=6, pady=4)

        self.Zf_var = tk.DoubleVar(value=0.0)
        self.Xd_pp_var = tk.DoubleVar(value=0.15)
        self.Xd_p_var = tk.DoubleVar(value=0.25)
        self.Xd_var = tk.DoubleVar(value=1.0)
        self.Ta_var = tk.DoubleVar(value=0.05)

        params = [
            ("Fault Z (Ω)", self.Zf_var, 0, 5),
            ("X''d (pu)", self.Xd_pp_var, 0.05, 0.5),
            ("X'd (pu)", self.Xd_p_var, 0.1, 1.0),
            ("Xd (pu)", self.Xd_var, 0.5, 3.0),
            ("Ta (s)", self.Ta_var, 0.01, 0.3),
        ]
        for c, (name, var, lo, hi) in enumerate(params):
            f2 = ttk.Frame(ctrl)
            f2.grid(row=0, column=c, padx=6, pady=4)
            ttk.Label(f2, text=name).pack()
            lbl = ttk.Label(f2, text=f"{var.get():.3f}")
            lbl.pack()
            ttk.Scale(f2, from_=lo, to=hi, variable=var, orient='horizontal', length=130,
                      command=lambda v, l=lbl, vr=var: (l.config(text=f"{vr.get():.3f}"),
                                                         self._plot_fault())).pack()

        ttk.Button(ctrl, text="Plot Fault Current", command=self._plot_fault).grid(
            row=1, column=0, columnspan=5, pady=6)

        info = ttk.Label(ctrl, text="3-phase fault: i(t) = √2·[I'' e^(-t/T'') + (I'-I'')e^(-t/T') + (I-I')e^(-t/T)] + DC offset",
                         foreground='#555')
        info.grid(row=2, column=0, columnspan=5, sticky='w', padx=6)

        self._plot_fault()

    def _plot_fault(self):
        try:
            V1 = self._get_V1()
            Zf = self.Zf_var.get()
            Xd_pp = self.Xd_pp_var.get()
            Xd_p = self.Xd_p_var.get()
            Xd = self.Xd_var.get()
            Ta = self.Ta_var.get()
            # Proper motor base impedance: Zbase = V_phase / I1_rated
            pb_base = motor_params_at_slip(V1, self.freq.get(),
                                           self.R1.get(), self.R2p.get(),
                                           self.X1.get(), self.X2p.get(),
                                           self.Xm.get(), 0.0333, self.poles)
            I1_rated = abs(pb_base['I1'])
            Zbase = V1 / I1_rated if I1_rated > 0 else max(1.0, abs(self.X1.get() + self.R1.get()))

            I_pp = V1 / (Xd_pp * Zbase + Zf) if (Xd_pp * Zbase + Zf) > 0 else 0
            I_p = V1 / (Xd_p * Zbase + Zf) if (Xd_p * Zbase + Zf) > 0 else 0
            I_ss = V1 / (Xd * Zbase + Zf) if (Xd * Zbase + Zf) > 0 else 0
            T_pp = 0.03; T_p = 0.3

            t = np.linspace(0, 0.5, 2000)
            f_val = self.freq.get()
            envelope = (I_pp * np.exp(-t / T_pp) +
                        (I_p - I_pp) * np.exp(-t / T_p) +
                        I_ss)
            dc_offset = I_pp * np.exp(-t / Ta)
            i_ac = np.sqrt(2) * envelope * np.sin(2 * np.pi * f_val * t)
            i_total = i_ac + np.sqrt(2) * dc_offset

            fig = self._fault_fig
            fig.clf()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            ax1.plot(t, i_total, 'r-', linewidth=1, label='Total fault current', alpha=0.8)
            ax1.plot(t, np.sqrt(2) * envelope, 'b--', linewidth=1.5, label='AC envelope')
            ax1.plot(t, -np.sqrt(2) * envelope, 'b--', linewidth=1.5)
            ax1.set_title("3-Phase Short-Circuit Fault Current"); ax1.set_ylabel("Current (A)")
            ax1.legend(); ax1.grid(True, alpha=0.4)

            ax2.axhline(I_pp, color='r', ls='--', label=f"I'' (subtransient) = {I_pp:.2f}A")
            ax2.axhline(I_p, color='orange', ls='--', label=f"I' (transient) = {I_p:.2f}A")
            ax2.axhline(I_ss, color='g', ls='--', label=f"I (steady-state) = {I_ss:.2f}A")
            ax2.plot(t, envelope, 'b-', linewidth=2)
            ax2.set_title("RMS Envelope Decay"); ax2.set_ylabel("Current RMS (A)")
            ax2.set_xlabel("Time (s)"); ax2.legend(); ax2.grid(True, alpha=0.4)

            fig.tight_layout()
            self._fault_canvas.draw()
            self.status_var.set(f"Fault currents: I''={I_pp:.2f}A  I'={I_p:.2f}A  Iss={I_ss:.2f}A")
        except Exception as e:
            self.status_var.set(f"Fault plot error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 – Protection Coordination
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_protection(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" Protection ")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        fig = Figure(figsize=(10, 6), dpi=96, facecolor='#f8f8f8')
        self._prot_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=tab)
        self._prot_canvas = canvas
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        ctrl = ttk.LabelFrame(tab, text="Relay Settings (IDMT – Inverse Definite Minimum Time)")
        ctrl.grid(row=1, column=0, sticky='ew', padx=6, pady=4)

        self.Ip1 = tk.DoubleVar(value=1.0)
        self.TMS1 = tk.DoubleVar(value=0.1)
        self.Ip2 = tk.DoubleVar(value=0.5)
        self.TMS2 = tk.DoubleVar(value=0.3)

        relay_params = [
            ("Primary Ip (×In)", self.Ip1, 0.1, 2.0),
            ("Primary TMS", self.TMS1, 0.05, 1.0),
            ("Backup Ip (×In)", self.Ip2, 0.1, 2.0),
            ("Backup TMS", self.TMS2, 0.05, 1.5),
        ]
        for c, (name, var, lo, hi) in enumerate(relay_params):
            f2 = ttk.Frame(ctrl)
            f2.grid(row=0, column=c, padx=8, pady=4)
            ttk.Label(f2, text=name).pack()
            lbl = ttk.Label(f2, text=f"{var.get():.3f}")
            lbl.pack()
            ttk.Scale(f2, from_=lo, to=hi, variable=var, orient='horizontal', length=150,
                      command=lambda v, l=lbl, vr=var: (l.config(text=f"{vr.get():.3f}"),
                                                         self._plot_protection())).pack()

        info = ttk.Label(ctrl, text="IEC Standard Inverse: t = TMS × 0.14 / [(I/Ip)^0.02 - 1]  |  "
                                    "CTI (Coordination Time Interval) = 0.3s minimum",
                         foreground='#555')
        info.grid(row=1, column=0, columnspan=4, sticky='w', padx=6)
        self._plot_protection()

    def _plot_protection(self):
        try:
            fig = self._prot_fig
            fig.clf()
            ax = fig.add_subplot(111)
            I_mult = np.logspace(0.1, 2.0, 500)

            def idmt(I, Ip, TMS, std='SI'):
                M = I / Ip
                M = np.where(M > 1.0, M, np.nan)
                if std == 'SI':
                    return TMS * 0.14 / (M ** 0.02 - 1)
                elif std == 'VI':
                    return TMS * 13.5 / (M - 1)
                else:
                    return TMS * 80 / (M ** 2 - 1)

            Ip1 = self.Ip1.get(); TMS1 = self.TMS1.get()
            Ip2 = self.Ip2.get(); TMS2 = self.TMS2.get()

            t1_SI = idmt(I_mult, Ip1, TMS1, 'SI')
            t2_SI = idmt(I_mult, Ip2, TMS2, 'SI')
            t1_VI = idmt(I_mult, Ip1, TMS1, 'VI')
            t2_VI = idmt(I_mult, Ip2, TMS2, 'VI')

            ax.loglog(I_mult, t1_SI, 'b-', linewidth=2, label=f'Primary SI (Ip={Ip1:.2f},TMS={TMS1:.2f})')
            ax.loglog(I_mult, t2_SI, 'r-', linewidth=2, label=f'Backup SI (Ip={Ip2:.2f},TMS={TMS2:.2f})')
            ax.loglog(I_mult, t1_VI, 'b--', linewidth=1.5, alpha=0.7, label='Primary VI')
            ax.loglog(I_mult, t2_VI, 'r--', linewidth=1.5, alpha=0.7, label='Backup VI')

            ax.axhline(0.3, color='g', ls=':', label='Min CTI=0.3s')
            ax.fill_between(I_mult, t1_SI, t2_SI, alpha=0.1, color='yellow',
                            where=~np.isnan(t1_SI) & ~np.isnan(t2_SI), label='Coordination zone')

            ax.set_xlabel("Current (× pickup)", fontsize=11)
            ax.set_ylabel("Operating Time (s)", fontsize=11)
            ax.set_title("Time-Current Characteristics – IDMT Protection Coordination", fontsize=12)
            ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
            ax.set_xlim(1, 100); ax.set_ylim(0.01, 100)
            fig.tight_layout()
            self._prot_canvas.draw()
        except Exception as e:
            self.status_var.set(f"Protection plot error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 – Speed Controller
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_speed_ctrl(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" Speed Controller ")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        fig = Figure(figsize=(12, 7), dpi=96, facecolor='#f8f8f8')
        self._ctrl_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=tab)
        self._ctrl_canvas = canvas
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        ctrl = ttk.LabelFrame(tab, text="Controller Parameters")
        ctrl.grid(row=1, column=0, sticky='ew', padx=6, pady=4)

        self.Kp_var = tk.DoubleVar(value=2.0)
        self.Ki_var = tk.DoubleVar(value=0.5)
        self.Kd_var = tk.DoubleVar(value=0.1)
        self.ref_speed_var = tk.DoubleVar(value=1450.0)
        self.step_TL_var = tk.DoubleVar(value=30.0)
        self.step_t_var = tk.DoubleVar(value=2.0)

        sp = [
            ("Kp", self.Kp_var, 0, 20),
            ("Ki", self.Ki_var, 0, 5),
            ("Kd", self.Kd_var, 0, 2),
            ("Ref Speed (rpm)", self.ref_speed_var, 100, 1500),
            ("Step Load (N·m)", self.step_TL_var, 0, 100),
            ("Step time (s)", self.step_t_var, 0.5, 8),
        ]
        for c, (name, var, lo, hi) in enumerate(sp):
            f2 = ttk.Frame(ctrl)
            f2.grid(row=0, column=c, padx=6, pady=2)
            ttk.Label(f2, text=name).pack()
            lbl = ttk.Label(f2, text=f"{var.get():.2f}")
            lbl.pack()
            ttk.Scale(f2, from_=lo, to=hi, variable=var, orient='horizontal', length=130,
                      command=lambda v, l=lbl, vr=var: l.config(text=f"{vr.get():.2f}")).pack()

        btnf = ttk.Frame(ctrl)
        btnf.grid(row=1, column=0, columnspan=6, pady=4)
        ttk.Button(btnf, text="▶ Run", command=self._run_speed_ctrl).pack(side='left', padx=4)
        ttk.Button(btnf, text="⏹ Stop", command=lambda: setattr(self, 'speed_ctrl_running', False)).pack(side='left', padx=4)
        ttk.Button(btnf, text="↺ Reset", command=self._reset_speed_ctrl).pack(side='left', padx=4)

        info = ttk.Label(ctrl, text="PID: u(t)=Kp·e+Ki·∫e dt+Kd·de/dt  |  Fuzzy: rule-base with e and Δe as inputs",
                         foreground='#555')
        info.grid(row=2, column=0, columnspan=6, sticky='w', padx=6)

    def _run_speed_ctrl(self):
        try:
            V1 = self._get_V1(); f = self.freq.get()
            R1 = self.R1.get(); R2p = self.R2p.get()
            X1 = self.X1.get(); X2p = self.X2p.get()
            Xm = self.Xm.get(); poles = self.poles
            J = self.J_var.get()
            omega_s = 2 * np.pi * f / (poles / 2)
            ref_speed = self.ref_speed_var.get() * 2 * np.pi / 60
            step_TL = self.step_TL_var.get()
            step_t = self.step_t_var.get()
            Kp = self.Kp_var.get(); Ki = self.Ki_var.get(); Kd = self.Kd_var.get()
            T = 10.0
            dt = 0.02

            # PID simulation
            t_arr = np.arange(0, T, dt)
            omega_pid = np.zeros_like(t_arr)
            Te_cmd_pid = np.zeros_like(t_arr)
            integral = 0.0; prev_err = 0.0
            omega_pid[0] = 0.0

            for i in range(1, len(t_arr)):
                TL = step_TL if t_arr[i] >= step_t else 5.0
                err = ref_speed - omega_pid[i - 1]
                integral += err * dt
                deriv = (err - prev_err) / dt
                u = Kp * err + Ki * integral + Kd * deriv
                u = max(0, min(u, 200))
                prev_err = err
                s = max(1e-6, min(1.0, (omega_s - omega_pid[i - 1]) / omega_s))
                try:
                    Te_base, _ = compute_torque(V1, f, R1, R2p, X1, X2p, Xm, s, poles)
                    Te = min(Te_base, u)
                except:
                    Te = 0
                Te_cmd_pid[i] = Te
                domega = (Te - TL) / J
                omega_pid[i] = max(0, omega_pid[i - 1] + domega * dt)

            speed_pid = omega_pid * 60 / (2 * np.pi)

            # Fuzzy simulation (simplified Mamdani)
            omega_fuzzy = np.zeros_like(t_arr)
            omega_fuzzy[0] = 0.0

            def fuzzy_output(e, de):
                NB, NM, Z, PM, PB = -200, -100, 0, 100, 200
                if e > 50:
                    if de > 10: return 180
                    elif de > -10: return 140
                    else: return 100
                elif e > 10:
                    if de > 10: return 120
                    elif de > -10: return 80
                    else: return 50
                elif e > -10:
                    if de > 10: return 60
                    elif de > -10: return 30
                    else: return 10
                else:
                    if de > 10: return 20
                    else: return 5

            prev_e = 0.0
            for i in range(1, len(t_arr)):
                TL = step_TL if t_arr[i] >= step_t else 5.0
                e = (ref_speed - omega_fuzzy[i - 1]) * 60 / (2 * np.pi)
                de = (e - prev_e) / dt
                prev_e = e
                u_fuzzy = fuzzy_output(e, de)
                s = max(1e-6, min(1.0, (omega_s - omega_fuzzy[i - 1]) / omega_s))
                try:
                    Te_base, _ = compute_torque(V1, f, R1, R2p, X1, X2p, Xm, s, poles)
                    Te = min(Te_base, u_fuzzy)
                except:
                    Te = 0
                domega = (Te - TL) / J
                omega_fuzzy[i] = max(0, omega_fuzzy[i - 1] + domega * dt)

            speed_fuzzy = omega_fuzzy * 60 / (2 * np.pi)
            ref_arr = np.full_like(t_arr, self.ref_speed_var.get())

            fig = self._ctrl_fig
            fig.clf()
            gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])

            ax1.plot(t_arr, speed_pid, 'b-', linewidth=2, label='PID speed')
            ax1.plot(t_arr, ref_arr, 'r--', label='Reference')
            ax1.axvline(step_t, color='g', ls=':', label=f'Load step@{step_t}s')
            ax1.set_title("PID Speed Response"); ax1.set_ylabel("Speed (rpm)")
            ax1.set_xlabel("Time (s)"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4)

            ax2.plot(t_arr, speed_fuzzy, 'm-', linewidth=2, label='Fuzzy speed')
            ax2.plot(t_arr, ref_arr, 'r--', label='Reference')
            ax2.axvline(step_t, color='g', ls=':', label=f'Load step@{step_t}s')
            ax2.set_title("Fuzzy Speed Response"); ax2.set_ylabel("Speed (rpm)")
            ax2.set_xlabel("Time (s)"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4)

            ax3.plot(t_arr, speed_pid, 'b-', linewidth=2, label='PID')
            ax3.plot(t_arr, speed_fuzzy, 'm-', linewidth=2, label='Fuzzy')
            ax3.plot(t_arr, ref_arr, 'r--', label='Reference')
            ax3.set_title("Comparison"); ax3.set_ylabel("Speed (rpm)")
            ax3.set_xlabel("Time (s)"); ax3.legend(fontsize=8); ax3.grid(True, alpha=0.4)

            # Fuzzy membership function illustration
            e_range = np.linspace(-200, 200, 500)
            def trimf(x, a, b, c):
                return np.maximum(0, np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)))

            ax4.plot(e_range, trimf(e_range, -200, -100, 0), 'r-', label='NB')
            ax4.plot(e_range, trimf(e_range, -100, 0, 100), 'orange', label='NM/Z')
            ax4.plot(e_range, trimf(e_range, 0, 100, 200), 'g-', label='PB')
            ax4.set_title("Fuzzy Membership Functions (Error)"); ax4.set_xlabel("Error (rpm)")
            ax4.set_ylabel("Membership"); ax4.legend(fontsize=8); ax4.grid(True, alpha=0.4)

            self._ctrl_canvas.draw()
            self.status_var.set("Speed control simulation complete.")
        except Exception as e:
            self.status_var.set(f"Speed ctrl error: {e}")

    def _reset_speed_ctrl(self):
        self._ctrl_fig.clf()
        ax = self._ctrl_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Press 'Run' to simulate PID and Fuzzy speed control",
                ha='center', va='center', transform=ax.transAxes, fontsize=13, color='gray')
        self._ctrl_canvas.draw()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 – Thermal Analysis
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_thermal(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" Thermal ")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        fig = Figure(figsize=(10, 6), dpi=96, facecolor='#f8f8f8')
        self._therm_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=tab)
        self._therm_canvas = canvas
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        ctrl = ttk.LabelFrame(tab, text="Thermal Model Parameters")
        ctrl.grid(row=1, column=0, sticky='ew', padx=6, pady=4)

        self.T_amb = tk.DoubleVar(value=25.0)
        self.Rth_var = tk.DoubleVar(value=0.15)
        self.Cth_var = tk.DoubleVar(value=2000.0)
        self.load_pct_var = tk.DoubleVar(value=100.0)

        th_params = [
            ("Ambient T (°C)", self.T_amb, 0, 50),
            ("Rth (°C/W)", self.Rth_var, 0.01, 1.0),
            ("Cth (J/°C)", self.Cth_var, 100, 10000),
            ("Load (%)", self.load_pct_var, 10, 125),
        ]
        for c, (name, var, lo, hi) in enumerate(th_params):
            f2 = ttk.Frame(ctrl)
            f2.grid(row=0, column=c, padx=8, pady=4)
            ttk.Label(f2, text=name).pack()
            lbl = ttk.Label(f2, text=f"{var.get():.2f}")
            lbl.pack()
            ttk.Scale(f2, from_=lo, to=hi, variable=var, orient='horizontal', length=160,
                      command=lambda v, l=lbl, vr=var: (l.config(text=f"{vr.get():.2f}"),
                                                         self._plot_thermal())).pack()

        info = ttk.Label(ctrl, text="Thermal model: Cth·dT/dt = Ploss - (T-Tamb)/Rth  |  T_steady = Tamb + Ploss·Rth",
                         foreground='#555')
        info.grid(row=1, column=0, columnspan=4, sticky='w', padx=6)
        self._plot_thermal()

    def _plot_thermal(self):
        try:
            V1 = self._get_V1(); f = self.freq.get()
            R1 = self.R1.get(); R2p = self.R2p.get()
            X1 = self.X1.get(); X2p = self.X2p.get()
            Xm = self.Xm.get(); Prot = self.Prot.get()
            poles = self.poles; s_fl = 0.0333
            T_amb = self.T_amb.get()
            Rth = self.Rth_var.get(); Cth = self.Cth_var.get()
            load = self.load_pct_var.get() / 100.0

            def thermal_sim(load_factor):
                pb = motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm,
                                          s_fl, poles, Prot)
                I1 = abs(pb['I1']) * load_factor
                I2p = abs(pb['I2p']) * load_factor
                P_cu1 = 3 * I1**2 * R1
                P_cu2 = 3 * I2p**2 * R2p
                P_fe = 0.02 * V1**2 / Xm
                Ploss = P_cu1 + P_cu2 + P_fe + Prot * 0.3

                t = np.linspace(0, 7200, 3600)
                tau = Rth * Cth
                T_inf = T_amb + Ploss * Rth
                T = T_inf - (T_inf - T_amb) * np.exp(-t / tau)
                return t, T, Ploss, T_inf

            fig = self._therm_fig
            fig.clf()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)

            load_levels = [0.25, 0.50, 0.75, 1.0, load]
            colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
            labels = ['25%', '50%', '75%', '100%', f'{load*100:.0f}% (slider)']
            seen = set()
            for lf, col, lab in zip(load_levels, colors, labels):
                if lf in seen:
                    continue
                seen.add(lf)
                t, T, Ploss, T_inf = thermal_sim(lf)
                ax1.plot(t / 60, T, color=col, linewidth=2, label=f'{lab}: Ploss={Ploss:.0f}W, T∞={T_inf:.1f}°C')

            ax1.axhline(105, color='r', ls='--', label='Limit 105°C (Class B)')
            ax1.set_xlabel("Time (min)"); ax1.set_ylabel("Winding Temp (°C)")
            ax1.set_title("Thermal Rise – Winding Temperature vs Time")
            ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4)

            loads = np.linspace(0.1, 1.25, 100)
            T_ss = []
            for lf in loads:
                _, _, Ploss, T_inf = thermal_sim(lf)
                T_ss.append(T_inf)
            ax2.plot(loads * 100, T_ss, 'b-', linewidth=2)
            ax2.axhline(105, color='r', ls='--', label='105°C limit')
            ax2.axhline(130, color='orange', ls='--', label='130°C limit (Class F)')
            ax2.set_xlabel("Load (%)"); ax2.set_ylabel("Steady-state Temp (°C)")
            ax2.set_title("Steady-State Temperature vs Load")
            ax2.legend(); ax2.grid(True, alpha=0.4)

            fig.tight_layout()
            self._therm_canvas.draw()
        except Exception as e:
            self.status_var.set(f"Thermal error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8 – Economic Analysis
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_economic(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" Economic ")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        fig = Figure(figsize=(11, 7), dpi=96, facecolor='#f8f8f8')
        self._econ_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=tab)
        self._econ_canvas = canvas
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        ctrl = ttk.LabelFrame(tab, text="Economic Parameters")
        ctrl.grid(row=1, column=0, sticky='ew', padx=6, pady=4)

        self.tariff_var = tk.DoubleVar(value=0.15)
        self.hours_var = tk.DoubleVar(value=4000.0)
        self.motor_cost_var = tk.DoubleVar(value=5000.0)
        self.eff_new_var = tk.DoubleVar(value=94.0)

        eco_params = [
            ("Tariff ($/kWh)", self.tariff_var, 0.05, 0.50),
            ("Hours/year", self.hours_var, 1000, 8760),
            ("New motor cost ($)", self.motor_cost_var, 500, 20000),
            ("New motor η (%)", self.eff_new_var, 88, 98),
        ]
        for c, (name, var, lo, hi) in enumerate(eco_params):
            f2 = ttk.Frame(ctrl)
            f2.grid(row=0, column=c, padx=8, pady=4)
            ttk.Label(f2, text=name).pack()
            lbl = ttk.Label(f2, text=f"{var.get():.2f}")
            lbl.pack()
            ttk.Scale(f2, from_=lo, to=hi, variable=var, orient='horizontal', length=160,
                      command=lambda v, l=lbl, vr=var: (l.config(text=f"{vr.get():.2f}"),
                                                         self._plot_economic())).pack()

        self._plot_economic()

    def _plot_economic(self):
        try:
            V1 = self._get_V1(); f = self.freq.get()
            R1 = self.R1.get(); R2p = self.R2p.get()
            X1 = self.X1.get(); X2p = self.X2p.get()
            Xm = self.Xm.get(); Prot = self.Prot.get()
            poles = self.poles; s_fl = 0.0333
            tariff = self.tariff_var.get()
            hours = self.hours_var.get()
            cost_new = self.motor_cost_var.get()
            eta_new = self.eff_new_var.get() / 100

            loads = [0.25, 0.50, 0.75, 1.00]
            labels = ['25%', '50%', '75%', '100%']
            etas = []; Pouts = []; Pins = []; annual_old = []; annual_new = []

            for lf in loads:
                pb = motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm, s_fl, poles, Prot)
                eta = pb['eta_overall']
                Pout = pb['Pout'] * lf
                Pin_old = Pout / max(eta * lf, 1e-3) if eta > 0 else Pout
                if lf < 1.0:
                    I_lf = abs(pb['I1']) * lf
                    P_cu1 = 3 * I_lf**2 * R1
                    P_cu2 = 3 * (abs(pb['I2p']) * lf)**2 * R2p
                    P_fe = 0.02 * V1**2 / Xm
                    Ploss = P_cu1 + P_cu2 + P_fe + Prot
                    Pout_lf = Pout - Prot * (1 - lf)
                    Pin_old = Pout_lf + Ploss if Pout_lf > 0 else pb['Pin'] * lf
                    eta = max(0, Pout_lf / Pin_old) if Pin_old > 0 else 0
                else:
                    Pin_old = pb['Pin']
                    eta = pb['eta_overall']
                    Pout = pb['Pout']

                Pin_new = Pout / max(eta_new, 1e-6)
                etas.append(eta * 100)
                Pouts.append(Pout / 1000)
                Pins.append(Pin_old / 1000)
                annual_old.append(Pin_old / 1000 * hours * tariff)
                annual_new.append(Pin_new / 1000 * hours * tariff)

            savings = [a - b for a, b in zip(annual_old, annual_new)]
            payback = [cost_new / max(s, 0.01) for s in savings]

            fig = self._econ_fig
            fig.clf()
            gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])

            x = np.arange(len(loads))
            ax1.bar(x, etas, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'], width=0.6)
            ax1.set_xticks(x); ax1.set_xticklabels(labels)
            ax1.set_ylabel("Efficiency (%)"); ax1.set_title("Motor Efficiency vs Load")
            ax1.set_ylim(0, 100); ax1.grid(True, axis='y', alpha=0.4)
            for i, v in enumerate(etas):
                ax1.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=9)

            w = 0.35
            ax2.bar(x - w/2, Pins, w, label='Input kW (old motor)', color='#F44336', alpha=0.8)
            ax2.bar(x + w/2, Pouts, w, label='Output kW', color='#4CAF50', alpha=0.8)
            ax2.set_xticks(x); ax2.set_xticklabels(labels)
            ax2.set_ylabel("Power (kW)"); ax2.set_title("Input vs Output Power")
            ax2.legend(fontsize=8); ax2.grid(True, axis='y', alpha=0.4)

            ax3.bar(x - w/2, annual_old, w, label='Old motor', color='#F44336', alpha=0.8)
            ax3.bar(x + w/2, annual_new, w, label=f'New motor η={eta_new*100:.0f}%', color='#4CAF50', alpha=0.8)
            ax3.set_xticks(x); ax3.set_xticklabels(labels)
            ax3.set_ylabel("Annual Cost ($)"); ax3.set_title(f"Annual Energy Cost ({hours:.0f}h @ ${tariff}/kWh)")
            ax3.legend(fontsize=8); ax3.grid(True, axis='y', alpha=0.4)

            ax4.bar(x, payback, color='#9C27B0', width=0.6)
            ax4.set_xticks(x); ax4.set_xticklabels(labels)
            ax4.set_ylabel("Payback Period (years)"); ax4.set_title(f"Payback Period (Motor cost=${cost_new:.0f})")
            ax4.grid(True, axis='y', alpha=0.4)
            for i, v in enumerate(payback):
                ax4.text(i, v + 0.05, f"{v:.1f}yr", ha='center', fontsize=9)

            fig.tight_layout()
            self._econ_canvas.draw()
        except Exception as e:
            self.status_var.set(f"Economic error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 9 – Harmonic Analysis
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_harmonics(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" Harmonics ")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)

        fig = Figure(figsize=(11, 7), dpi=96, facecolor='#f8f8f8')
        self._harm_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=tab)
        self._harm_canvas = canvas
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        ctrl = ttk.LabelFrame(tab, text="Harmonic Content (% of fundamental)")
        ctrl.grid(row=1, column=0, sticky='ew', padx=6, pady=4)

        self.h3_var = tk.DoubleVar(value=2.0)
        self.h5_var = tk.DoubleVar(value=8.0)
        self.h7_var = tk.DoubleVar(value=5.0)
        self.h11_var = tk.DoubleVar(value=3.0)
        self.h13_var = tk.DoubleVar(value=2.0)
        self.h_fund_var = tk.DoubleVar(value=10.0)

        harm_params = [
            ("I1 fund (A)", self.h_fund_var, 1, 50),
            ("3rd (%)", self.h3_var, 0, 20),
            ("5th (%)", self.h5_var, 0, 30),
            ("7th (%)", self.h7_var, 0, 25),
            ("11th (%)", self.h11_var, 0, 15),
            ("13th (%)", self.h13_var, 0, 10),
        ]
        for c, (name, var, lo, hi) in enumerate(harm_params):
            f2 = ttk.Frame(ctrl)
            f2.grid(row=0, column=c, padx=6, pady=4)
            ttk.Label(f2, text=name).pack()
            lbl = ttk.Label(f2, text=f"{var.get():.1f}")
            lbl.pack()
            ttk.Scale(f2, from_=lo, to=hi, variable=var, orient='horizontal', length=130,
                      command=lambda v, l=lbl, vr=var: (l.config(text=f"{vr.get():.1f}"),
                                                         self._plot_harmonics())).pack()

        self._plot_harmonics()

    def _plot_harmonics(self):
        try:
            f0 = self.freq.get()
            I1 = self.h_fund_var.get()
            h3 = self.h3_var.get() / 100 * I1
            h5 = self.h5_var.get() / 100 * I1
            h7 = self.h7_var.get() / 100 * I1
            h11 = self.h11_var.get() / 100 * I1
            h13 = self.h13_var.get() / 100 * I1

            Fs = 10000; N = 8192
            t = np.linspace(0, N / Fs, N, endpoint=False)
            i_wave = (I1 * np.sqrt(2) * np.sin(2 * np.pi * f0 * t) +
                      h3 * np.sqrt(2) * np.sin(2 * np.pi * 3 * f0 * t) +
                      h5 * np.sqrt(2) * np.sin(2 * np.pi * 5 * f0 * t) +
                      h7 * np.sqrt(2) * np.sin(2 * np.pi * 7 * f0 * t) +
                      h11 * np.sqrt(2) * np.sin(2 * np.pi * 11 * f0 * t) +
                      h13 * np.sqrt(2) * np.sin(2 * np.pi * 13 * f0 * t))

            freqs = fftfreq(N, 1 / Fs)
            spectrum = np.abs(fft(i_wave)) / N * 2
            THD = np.sqrt(h3**2 + h5**2 + h7**2 + h11**2 + h13**2) / I1 * 100

            fig = self._harm_fig
            fig.clf()
            gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])

            cycles = 3
            t_show = t[t < cycles / f0]
            ax1.plot(t_show * 1000, i_wave[:len(t_show)], 'b-', linewidth=1.5)
            ax1.set_xlabel("Time (ms)"); ax1.set_ylabel("Current (A)")
            ax1.set_title("Motor Current Waveform (with harmonics)")
            ax1.grid(True, alpha=0.4)

            pos_mask = (freqs >= 0) & (freqs <= 1500)
            ax2.stem(freqs[pos_mask], spectrum[pos_mask], linefmt='b-', markerfmt='bo',
                     basefmt='k-', use_line_collection=True)
            ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("|I| (A)")
            ax2.set_title(f"FFT Spectrum  (THD={THD:.2f}%)")
            ax2.grid(True, alpha=0.4)

            harm_orders = [1, 3, 5, 7, 11, 13]
            harm_amps = [I1, h3, h5, h7, h11, h13]
            harm_pct = [100, self.h3_var.get(), self.h5_var.get(),
                        self.h7_var.get(), self.h11_var.get(), self.h13_var.get()]
            ax3.bar([str(h) for h in harm_orders], harm_amps,
                    color=['#2196F3', '#F44336', '#FF9800', '#4CAF50', '#9C27B0', '#00BCD4'])
            ax3.set_xlabel("Harmonic Order"); ax3.set_ylabel("Amplitude (A)")
            ax3.set_title("Harmonic Bar Chart"); ax3.grid(True, axis='y', alpha=0.4)
            for i, (h, v) in enumerate(zip(harm_amps, harm_pct)):
                ax3.text(i, h + 0.1, f"{v:.1f}%", ha='center', fontsize=8)

            pb_fl = motor_params_at_slip(self._get_V1(), self.freq.get(),
                                         self.R1.get(), self.R2p.get(),
                                         self.X1.get(), self.X2p.get(),
                                         self.Xm.get(), 0.0333, self.poles)
            dpf = abs(pb_fl['pf'])
            indices = {'IHD': harm_pct[1:],
                       'THD': [THD],
                       'DPF': [dpf],
                       'PF_dist': [dpf / np.sqrt(1 + (THD / 100)**2)]}

            metrics = [f"THD = {THD:.2f} %",
                       f"5th = {self.h5_var.get():.1f} %  (limit 6%)",
                       f"7th = {self.h7_var.get():.1f} %  (limit 5%)",
                       f"11th = {self.h11_var.get():.1f} %  (limit 3.5%)",
                       f"13th = {self.h13_var.get():.1f} %  (limit 3%)",
                       f"PF_dist = {1/np.sqrt(1+(THD/100)**2):.4f}",
                       f"Crest factor = {np.max(np.abs(i_wave)) / (I1*np.sqrt(2)):.3f}",
                       f"IEC 61000-3-12 limits shown"]
            ax4.axis('off')
            ax4.set_title("Power Quality Indices")
            for i, m in enumerate(metrics):
                col = 'red' if ('THD' in m and THD > 8) else 'green'
                ax4.text(0.05, 0.9 - i * 0.1, m, transform=ax4.transAxes,
                         fontsize=10, color=col, family='monospace')

            fig.tight_layout()
            self._harm_canvas.draw()
            self.status_var.set(f"THD = {THD:.2f}%")
        except Exception as e:
            self.status_var.set(f"Harmonic error: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 10 – Comprehensive Analysis
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab_comprehensive(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text=" Comprehensive ")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=3)
        tab.columnconfigure(1, weight=1)

        fig = Figure(figsize=(12, 8), dpi=96, facecolor='#f8f8f8')
        self._comp_fig = fig
        canvas = FigureCanvasTkAgg(fig, master=tab)
        self._comp_canvas = canvas
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        # Summary text
        rf = ttk.LabelFrame(tab, text="Summary Table")
        rf.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        rf.rowconfigure(0, weight=1); rf.columnconfigure(0, weight=1)
        self._comp_text = tk.Text(rf, wrap='word', font=('Courier', 9),
                                   bg='#1e1e1e', fg='#d4d4d4', width=32)
        self._comp_text.grid(row=0, column=0, sticky='nsew')

        ctrl = ttk.LabelFrame(tab, text="Display Options")
        ctrl.grid(row=1, column=0, columnspan=2, sticky='ew', padx=6, pady=4)
        ttk.Button(ctrl, text="Refresh All", command=self._plot_comprehensive).pack(side='left', padx=6, pady=4)
        ttk.Label(ctrl, text="Shows: Efficiency map (η vs speed & load), Circle diagram, Operating summary",
                  foreground='#555').pack(side='left', padx=10)

        tab.bind('<Configure>', lambda e: self._plot_comprehensive())
        self._plot_comprehensive()

    def _plot_comprehensive(self):
        try:
            V1 = self._get_V1(); f = self.freq.get()
            R1 = self.R1.get(); R2p = self.R2p.get()
            X1 = self.X1.get(); X2p = self.X2p.get()
            Xm = self.Xm.get(); Prot = self.Prot.get()
            poles = self.poles
            ns = 120 * f / poles

            fig = self._comp_fig
            fig.clf()
            gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])

            # (1) Efficiency map
            slips = np.linspace(0.005, 0.20, 40)
            loads = np.linspace(0.2, 1.2, 30)
            eta_map = np.zeros((len(loads), len(slips)))
            speed_arr = ns * (1 - slips)
            for i, lf in enumerate(loads):
                for j, s in enumerate(slips):
                    try:
                        pb = motor_params_at_slip(V1 * lf, f, R1, R2p, X1, X2p, Xm, s, poles, Prot)
                        eta_map[i, j] = pb['eta_overall'] * 100
                    except:
                        eta_map[i, j] = 0
            eta_map = np.clip(eta_map, 0, 100)
            im = ax1.contourf(speed_arr, loads * 100, eta_map, levels=20, cmap='RdYlGn')
            fig.colorbar(im, ax=ax1, label='η (%)')
            ax1.set_xlabel("Speed (rpm)"); ax1.set_ylabel("Load (%)")
            ax1.set_title("Efficiency Map (η % vs Speed & Load)")

            # Mark rated point
            s_fl = 0.0333
            pb_fl = motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm, s_fl, poles, Prot)
            ax1.scatter([pb_fl['speed_rpm']], [100], color='white', s=100, zorder=5,
                        marker='*', label='Rated point')
            ax1.legend(fontsize=8)

            # (2) Circle diagram (Heyland)
            I_no_load, _, _, _ = equivalent_circuit(V1, f, R1, R2p, X1, X2p, Xm, s=0.001)
            I_sc, _, _, _ = equivalent_circuit(V1, f, R1, R2p, X1, X2p, Xm, s=1.0)

            slips_cd = np.linspace(0.001, 1.0, 200)
            I_circle = [equivalent_circuit(V1, f, R1, R2p, X1, X2p, Xm, s)[0] for s in slips_cd]
            Ix = [i.real for i in I_circle]
            Iy = [i.imag for i in I_circle]

            ax2.plot(Ix, Iy, 'b-', linewidth=2, label='Current locus')
            ax2.scatter([I_no_load.real], [I_no_load.imag], color='g', s=80, zorder=5, label='No-load')
            ax2.scatter([I_sc.real], [I_sc.imag], color='r', s=80, zorder=5, label='Short-circuit')
            ax2.scatter([pb_fl['I1'].real], [pb_fl['I1'].imag], color='orange', s=80, zorder=5,
                        label=f'FL (s={s_fl})')
            ax2.axhline(0, color='k', linewidth=0.5)
            ax2.axvline(0, color='k', linewidth=0.5)
            ax2.set_xlabel("Re(I1) →  Active component (A)")
            ax2.set_ylabel("Im(I1) →  Reactive component (A)")
            ax2.set_title("Heyland Circle Diagram")
            ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4)
            ax2.set_aspect('equal')

            # (3) Efficiency and pf vs load
            load_pct = np.linspace(10, 125, 100)
            etas_load = []; pfs_load = []; speeds_load = []
            for lp in load_pct:
                try:
                    lf = lp / 100
                    p = motor_params_at_slip(V1 * lf, f, R1, R2p, X1, X2p, Xm, s_fl, poles, Prot)
                    etas_load.append(p['eta_overall'] * 100)
                    pfs_load.append(p['pf'] * 100)
                    speeds_load.append(p['speed_rpm'])
                except:
                    etas_load.append(0); pfs_load.append(0); speeds_load.append(0)

            ln1, = ax3.plot(load_pct, etas_load, 'b-', linewidth=2, label='η (%)')
            ax3_r = ax3.twinx()
            ln2, = ax3_r.plot(load_pct, pfs_load, 'r--', linewidth=2, label='PF×100')
            ax3.set_xlabel("Load (%)"); ax3.set_ylabel("η (%)", color='b')
            ax3_r.set_ylabel("PF × 100", color='r')
            ax3.set_title("Efficiency & Power Factor vs Load")
            lines = [ln1, ln2]
            ax3.legend(lines, [l.get_label() for l in lines], fontsize=9)
            ax3.grid(True, alpha=0.4)

            # (4) Torque breakdown
            slips_tb = np.linspace(0.001, 1.0, 300)
            Te_arr = []; Tnet_arr = []
            for s in slips_tb:
                try:
                    p = motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm, s, poles, Prot)
                    Te_arr.append(p['Te'])
                    Tnet_arr.append(p['Tnet'])
                except:
                    Te_arr.append(0); Tnet_arr.append(0)

            speeds_tb = ns * (1 - slips_tb)
            ax4.plot(speeds_tb, Te_arr, 'b-', linewidth=2, label='Electromagnetic Te')
            ax4.plot(speeds_tb, Tnet_arr, 'g--', linewidth=2, label='Net Tnet (after Prot)')
            ax4.axhline(0, color='k', linewidth=0.5)
            ax4.set_xlabel("Speed (rpm)"); ax4.set_ylabel("Torque (N·m)")
            ax4.set_title("Torque Components vs Speed")
            ax4.legend(fontsize=9); ax4.grid(True, alpha=0.4)
            ax4.set_xlim(0, ns + 50)

            self._comp_canvas.draw()

            # Summary table
            pa = motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm, 1.0, poles, Prot)
            pb_fl2 = motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm, 0.0333, poles, Prot)
            s_mt = slip_max_torque(R1, R2p, X1, X2p, Xm)
            pc = motor_params_at_slip(V1, f, R1, R2p, X1, X2p, Xm, s_mt, poles, Prot)

            lines = [
                "COMPREHENSIVE SUMMARY",
                "=" * 28,
                f"Sync speed: {ns:.0f} rpm",
                f"V1 phase: {V1:.1f} V",
                "",
                "── Starting (s=1) ──────",
                f"I_start: {abs(pa['I1']):.2f} A",
                f"T_start: {pa['Te']:.2f} N·m",
                f"PF_start: {pa['pf']:.3f}",
                "",
                "── Full Load (s=0.0333) ─",
                f"Speed: {pb_fl2['speed_rpm']:.1f} rpm",
                f"I1: {abs(pb_fl2['I1']):.3f} A",
                f"PF: {pb_fl2['pf']:.4f}",
                f"Te: {pb_fl2['Te']:.2f} N·m",
                f"Tnet: {pb_fl2['Tnet']:.2f} N·m",
                f"Pin: {pb_fl2['Pin']/1000:.3f} kW",
                f"Pout: {pb_fl2['Pout']/1000:.3f} kW",
                f"η_int: {pb_fl2['eta_int']*100:.2f} %",
                f"η_ov: {pb_fl2['eta_overall']*100:.2f} %",
                "",
                "── Max Torque ──────────",
                f"s_mt: {s_mt:.4f}",
                f"Speed: {pc['speed_rpm']:.1f} rpm",
                f"T_max: {pc['Te']:.2f} N·m",
                f"T_max/T_FL: {pc['Te']/pb_fl2['Te']:.2f}",
                f"T_start/T_FL: {pa['Te']/pb_fl2['Te']:.2f}",
            ]
            self._comp_text.config(state='normal')
            self._comp_text.delete('1.0', 'end')
            self._comp_text.insert('end', "\n".join(lines))
            self._comp_text.config(state='disabled')
        except Exception as e:
            self.status_var.set(f"Comprehensive error: {e}")

    def _on_close(self):
        self.sim_running = False
        self.speed_ctrl_running = False
        try:
            plt.close('all')
        except:
            pass
        self.destroy()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = MotorAnalysisApp()
    app.mainloop()
