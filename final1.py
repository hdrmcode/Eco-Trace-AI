# main_gui.py  (final corrected version)
import cv2
import pandas as pd
import time
from collections import Counter
from difflib import get_close_matches
from ultralytics import YOLO
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import traceback
import re

# -------------------- Load YOLO model --------------------
MODEL_PATH = r"D:\Sprinthathon25\Main2\runs\retrain_from_last_40epochs\weights\best.pt"
model = None

try:
    if os.path.exists(MODEL_PATH):
        print("Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"⚠ Model file not found at: {MODEL_PATH}")
        model_dir = os.path.dirname(MODEL_PATH)
        if os.path.exists(model_dir):
            pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            if pt_files:
                alternative_path = os.path.join(model_dir, pt_files[0])
                print(f"Trying alternative model: {alternative_path}")
                model = YOLO(alternative_path)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    traceback.print_exc()

# -------------------- Load dataset --------------------
DATA_PATH = r"D:\Sprinthathon25\Main2\Filtered_Device_List.xlsx"
metal_data = None
device_names_list = []

try:
    if os.path.exists(DATA_PATH):
        print("Loading dataset...")
        metal_data = pd.read_excel(DATA_PATH)

        metal_data.columns = (metal_data.columns
                              .str.replace('\xa0', '', regex=False)
                              .str.strip()
                              .str.replace(' ', '_')
                              .str.lower())

        if 'normalized_name' not in metal_data.columns:
            print("⚠ 'normalized_name' column not found. Using first column instead.")
            first_col = metal_data.columns[0]
            metal_data['normalized_name'] = metal_data[first_col]

        device_names_list = metal_data['normalized_name'].astype(str).tolist()
        print(f"Dataset loaded with {len(device_names_list)} devices")
    else:
        print(f"⚠ Dataset file not found at: {DATA_PATH}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    traceback.print_exc()

# -------------------- Desired pie-chart columns (user-specified) --------------------
DESIRED_PIE_COLS_RAW = [
    "Plastic (%)",
    "Glass / Ceramic / Silicon (%)",
    "Metal (%)",
    "Other (%)",
    "Metal - Aluminum (%)",
    "Metal - Copper (%)",
    "Metal - Iron/Steel (%)",
    "Metal - Nickel (%)",
    "Metal - Tin (%)",
    "Metal - Gold (%)",
    "Metal - Silver (%)",
    "Metal - Palladium (%)",
    "Metal - Titanium (%)",
    "Other - Battery (%)",
    "Other - Rare Earths (%)",
    "Other - Solder (%)"
]


def _normalize_name(s: str) -> str:
    """Normalize column names for fuzzy matching."""
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9]', '', s)
    return s


def _is_metal_specific(desired_raw: str) -> bool:
    """Return True if the desired column represents a metal-specific category
    (Metal - Aluminum, Metal - Copper, Gold, Silver, etc.)."""
    if not desired_raw:
        return False
    low = desired_raw.lower()
    # if it's the generic "metal (%)", treat as overall metal, not metal-specific
    if re.sub(r'[^a-z0-9]', '', low) == _normalize_name("Metal (%)"):
        return False
    # metal-specific keys or metal names
    metal_keywords = [
        "metal-", "aluminum", "aluminium", "copper", "iron", "steel", "ironsteel",
        "nickel", "tin", "gold", "silver", "palladium", "titanium"
    ]
    for kw in metal_keywords:
        if kw in low:
            return True
    return False


# -------------------- Tkinter App --------------------
class EcoTraceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 EcoTrace AI - Smart Recycling Assistant")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f8f0")

        # -------------------- Camera --------------------
        self.cap = None
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("⚠ Could not open camera, trying alternative index...")
                self.cap = cv2.VideoCapture(1)
        except Exception as e:
            print(f"❌ Camera error: {e}")

        self.running = True if self.cap and self.cap.isOpened() else False

        # -------------------- Detection variables --------------------
        self.label_history = []
        self.detection_start_time = None
        self.fixed_label = None
        self.current_pie_data = None

        # -------------------- Stats --------------------
        self.items_recycled = 0
        self.wallet_balance = 0

        self.create_main_interface()

        # Start updating frames if camera is available
        if self.running:
            self.update_frame()
        else:
            self.update_result_text("❌ Camera not available!\n\nPlease check:\n• Camera is connected\n• No other app is using camera\n• Drivers are installed\n\nClick 'Restart Camera' to try again.")

    def create_main_interface(self):
        # Header Frame
        header_frame = Frame(self.root, bg="#2e7d32", height=120)
        header_frame.pack(fill="x", padx=10, pady=10)
        header_frame.pack_propagate(False)

        # Title
        self.title_label = Label(header_frame, text="🌿 EcoTrace AI", font=("Segoe UI", 28, "bold"),
                                 bg="#2e7d32", fg="white")
        self.title_label.pack(side=LEFT, padx=20, pady=20)

        # Impact Frame (right corner, placed)
        impact_frame = Frame(self.root, bg="#1b5e20", relief="raised", bd=3)
        impact_frame.place(relx=0.72, rely=0.12, relwidth=0.26, relheight=0.82)

        impact_title = Label(impact_frame, text="Your Impact",
                             font=("Segoe UI", 16, "bold"), bg="#1b5e20", fg="white")
        impact_title.pack(pady=(10, 5))

        # Chart frame (fixed size)
        self.chart_frame = Frame(impact_frame, bg="#1b5e20", width=260, height=260)
        self.chart_frame.pack(pady=(10, 5))
        self.chart_frame.pack_propagate(False)

        # Tree preview small
        self.tree_canvas_small = Canvas(impact_frame, width=220, height=160, bg="#1b5e20", highlightthickness=0)
        self.tree_canvas_small.pack(pady=(10, 5))

        # Separator
        separator = Frame(impact_frame, bg="#388e3c", height=2)
        separator.pack(fill=X, pady=5)

        # Bottom frame anchors items & wallet to bottom corner
        bottom_frame = Frame(impact_frame, bg="#1b5e20")
        bottom_frame.pack(side="bottom", fill="x", pady=(10, 15))
        bottom_frame.grid_columnconfigure(0, weight=1)
        bottom_frame.grid_columnconfigure(1, weight=1)

        # Items Recycled (left)
        self.items_label = Label(bottom_frame, text=f"Items Recycled\n{self.items_recycled}",
                                 font=("Segoe UI", 14, "bold"), bg="#1b5e20", fg="white", justify=CENTER)
        self.items_label.grid(row=0, column=0, sticky="w", padx=10)

        # Wallet Balance (right)
        self.wallet_label = Label(bottom_frame, text=f"Wallet Balance\n₹{self.wallet_balance}",
                                  font=("Segoe UI", 14, "bold"), bg="#1b5e20", fg="white", justify=CENTER)
        self.wallet_label.grid(row=0, column=1, sticky="e", padx=10)

        convertible_label = Label(impact_frame, text="* convertible to cash",
                                  font=("Segoe UI", 9), bg="#1b5e20", fg="#cccccc")
        convertible_label.pack(side="bottom", pady=(0, 10))

        # Status label in header
        self.status_label = Label(header_frame, text="Initializing...", font=("Segoe UI", 10),
                                  bg="#2e7d32", fg="white")
        self.status_label.pack(side=RIGHT, padx=20)
        self.update_status()

        # Main Content Area
        main_container = Frame(self.root, bg="#f0f8f0")
        main_container.pack(fill="both", expand=True, padx=20, pady=10)

        # Left Column - Camera and Results
        left_frame = Frame(main_container, bg="#f0f8f0")
        left_frame.pack(side=LEFT, fill="both", expand=True)

        # Camera Frame
        camera_frame = Frame(left_frame, bg="#2e7d32", relief="raised", bd=2)
        camera_frame.pack(fill="both", expand=True, pady=(0, 10))

        camera_title = Label(camera_frame, text="📷 Live Camera Feed", font=("Segoe UI", 16, "bold"),
                             bg="#2e7d32", fg="white", pady=10)
        camera_title.pack(fill="x")

        # Video feed with border
        video_container = Frame(camera_frame, bg="black", padx=2, pady=2)
        video_container.pack(padx=10, pady=10)

        self.video_label = Label(video_container, bg="black", width=640, height=480)
        self.video_label.pack()

        # Control Buttons
        control_frame = Frame(left_frame, bg="#f0f8f0")
        control_frame.pack(fill="x", pady=10)

        self.stop_btn = Button(control_frame, text="🛑 Stop Camera", command=self.stop_camera,
                               bg="#e53935", fg="white", font=("Segoe UI", 12, "bold"),
                               width=15, height=2)
        self.stop_btn.pack(side=LEFT, padx=5)

        self.restart_btn = Button(control_frame, text="🔄 Restart Camera", command=self.restart_camera,
                                  bg="#43a047", fg="white", font=("Segoe UI", 12, "bold"),
                                  width=15, height=2)
        self.restart_btn.pack(side=LEFT, padx=5)

        self.analytics_btn = Button(control_frame, text="📊 View Analytics", command=self.show_analytics,
                                    bg="#2196F3", fg="white", font=("Segoe UI", 12, "bold"),
                                    width=15, height=2)
        self.analytics_btn.pack(side=LEFT, padx=5)

        self.tree_btn = Button(control_frame, text="🌳 View EcoTree", command=self.show_tree,
                               bg="#FF9800", fg="white", font=("Segoe UI", 12, "bold"),
                               width=15, height=2)
        self.tree_btn.pack(side=LEFT, padx=5)

        # Results Frame
        results_frame = Frame(left_frame, bg="#ffffff", relief="sunken", bd=2)
        results_frame.pack(fill="both", expand=True)

        results_title = Label(results_frame, text="📊 Detection Results", font=("Segoe UI", 16, "bold"),
                              bg="#4caf50", fg="white", pady=10)
        results_title.pack(fill="x")

        # Create scrollable text area
        text_container = Frame(results_frame, bg="white")
        text_container.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar_text = Scrollbar(text_container)
        scrollbar_text.pack(side=RIGHT, fill=Y)

        self.result_text = Text(text_container,
                                wrap=WORD,
                                yscrollcommand=scrollbar_text.set,
                                font=("Segoe UI", 11),
                                bg="#f8fdf8",
                                fg="#2e7d32",
                                padx=15,
                                pady=15,
                                width=60,
                                height=15)
        self.result_text.pack(side=LEFT, fill="both", expand=True)

        initial_text = """🌿 Welcome to EcoTrace AI! 🌿

Your smart recycling assistant that:
• Detects electronic devices
• Analyzes metal composition
• Calculates recycling value
• Tracks your eco-impact

Point your camera at any electronic device to get started!"""

        self.result_text.insert(END, initial_text)
        self.result_text.config(state=DISABLED)
        scrollbar_text.config(command=self.result_text.yview)

        # Right Column - Quick Stats
        right_frame = Frame(main_container, bg="#f0f8f0")
        right_frame.pack(side=RIGHT, fill="both", expand=False, padx=(20, 0))

        # Quick Stats Frame
        stats_frame = Frame(right_frame, bg="#ffffff", relief="raised", bd=2)
        stats_frame.pack(fill="both", expand=True, pady=(0, 10))

        stats_title = Label(stats_frame, text="📈 Quick Stats", font=("Segoe UI", 16, "bold"),
                            bg="#4caf50", fg="white", pady=10)
        stats_title.pack(fill="x")

        stats_content = Frame(stats_frame, bg="white", padx=10, pady=10)
        stats_content.pack(fill="both", expand=True)

        # Current Device Info
        self.current_device_label = Label(stats_content, text="No device detected",
                                          font=("Segoe UI", 12, "bold"), bg="white", fg="#666")
        self.current_device_label.pack(anchor="w", pady=5)

        # Last Detection Value
        self.last_value_label = Label(stats_content, text="Last value: ₹0",
                                      font=("Segoe UI", 11), bg="white", fg="#2e7d32")
        self.last_value_label.pack(anchor="w", pady=2)

        # Materials Found
        self.materials_label = Label(stats_content, text="Materials: None",
                                     font=("Segoe UI", 10), bg="white", fg="#666", justify=LEFT)
        self.materials_label.pack(anchor="w", pady=5, fill="x")

        # Separator
        separator = Frame(stats_content, height=2, bg="#e0e0e0")
        separator.pack(fill="x", pady=10)

        analytics_preview = Label(stats_content, text="📊 Quick Access",
                                  font=("Segoe UI", 14, "bold"), bg="white", fg="#2e7d32")
        analytics_preview.pack(anchor="w", pady=5)

        preview_text = """Click buttons below to view:
• Material Composition Charts
• Your Growing EcoTree
• Detailed Analytics"""

        preview_label = Label(stats_content, text=preview_text, font=("Segoe UI", 10),
                              bg="white", fg="#666", justify=LEFT)
        preview_label.pack(anchor="w", pady=5, fill="x")

        # Tips Frame
        tips_frame = Frame(right_frame, bg="#ffffff", relief="raised", bd=2)
        tips_frame.pack(fill="both", expand=True)

        tips_title = Label(tips_frame, text="💡 Recycling Tips", font=("Segoe UI", 16, "bold"),
                           bg="#4caf50", fg="white", pady=10)
        tips_title.pack(fill="x")

        tips_content = Frame(tips_frame, bg="white", padx=10, pady=10)
        tips_content.pack(fill="both", expand=True)

        tips_text = """✅ Proper E-Waste Recycling:

1. Remove batteries before recycling
2. Wipe personal data from devices
3. Separate different material types
4. Use authorized recycling centers
5. Don't mix with regular trash

🌱 Every device recycled helps:
• Conserve natural resources
• Reduce landfill waste
• Prevent soil contamination"""

        tips_label = Label(tips_content, text=tips_text, font=("Segoe UI", 9),
                           bg="white", fg="#666", justify=LEFT)
        tips_label.pack(anchor="w", fill="both")

    def update_status(self):
        """Update the status label with current system state"""
        status_parts = []

        if model:
            status_parts.append("✅ Model Loaded")
        else:
            status_parts.append("❌ Model Failed")

        if metal_data is not None and len(device_names_list) > 0:
            status_parts.append("✅ Dataset Loaded")
        else:
            status_parts.append("❌ Dataset Failed")

        if self.cap and self.cap.isOpened():
            status_parts.append("✅ Camera Active")
        else:
            status_parts.append("❌ Camera Inactive")

        self.status_label.config(text=" | ".join(status_parts))

    def update_impact_display(self):
        """Update the impact display"""
        self.items_label.config(text=f"Items Recycled\n{self.items_recycled}")
        # show wallet with integer rupee value
        self.wallet_label.config(text=f"Wallet Balance\n₹{int(round(self.wallet_balance))}")

    def update_result_text(self, text):
        """Update the result text area with new content"""
        self.result_text.config(state=NORMAL)
        self.result_text.delete(1.0, END)
        self.result_text.insert(END, text)
        self.result_text.config(state=DISABLED)
        self.result_text.see(END)

    def update_quick_stats(self, device_name, value, materials):
        """Update the quick stats panel"""
        self.current_device_label.config(text=f"Device: {device_name}")
        self.last_value_label.config(text=f"Last value: ₹{int(round(value))}")

        materials_text = "Materials: " + ", ".join(materials.keys()) if materials else "Materials: None"
        self.materials_label.config(text=materials_text)

    # -------------------- Camera Control --------------------
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')
        self.update_status()

    def restart_camera(self):
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.running = True
                self.update_frame()
                self.update_result_text("✅ Camera restarted successfully!\n\nDetection is now running...\nPoint your camera at any electronic device to start analysis.")
            else:
                self.update_result_text("❌ Failed to restart camera.\n\nPlease check your camera connection and try again.")
        except Exception as e:
            self.update_result_text(f"❌ Camera restart error: {str(e)}")
        self.update_status()

    # -------------------- Analytics Window --------------------
    def show_analytics(self):
        """Show analytics in a separate window"""
        analytics_window = Toplevel(self.root)
        analytics_window.title("📊 EcoTrace Analytics - Material Composition")
        analytics_window.geometry("1000x700")
        analytics_window.configure(bg="#f0f8f0")
        analytics_window.transient(self.root)
        analytics_window.grab_set()

        # Header
        header = Frame(analytics_window, bg="#2e7d32", height=80)
        header.pack(fill="x", padx=10, pady=10)
        header.pack_propagate(False)

        title = Label(header, text="📊 Material Composition Analysis", font=("Segoe UI", 20, "bold"),
                      bg="#2e7d32", fg="white")
        title.pack(pady=20)

        # Content Frame
        content = Frame(analytics_window, bg="#f0f8f0")
        content.pack(fill="both", expand=True, padx=20, pady=10)

        if self.current_pie_data:
            self.create_professional_pie_chart(content, self.current_pie_data)
        else:
            no_data = Label(content, text="No device analyzed yet\n\nScan a device to see its material composition",
                            font=("Segoe UI", 14), bg="#f0f8f0", fg="#666", justify=CENTER)
            no_data.pack(expand=True)

        # Close button
        close_btn = Button(analytics_window, text="Close Analytics",
                           command=analytics_window.destroy,
                           bg="#e53935", fg="white", font=("Segoe UI", 12, "bold"),
                           width=15, height=2)
        close_btn.pack(pady=10)

    def show_tree(self):
        """Show tree in a separate window"""
        tree_window = Toplevel(self.root)
        tree_window.title("🌳 Your EcoTree Progress")
        tree_window.geometry("800x700")
        tree_window.configure(bg="#f0f8f0")
        tree_window.transient(self.root)
        tree_window.grab_set()

        # Header
        header = Frame(tree_window, bg="#2e7d32", height=80)
        header.pack(fill="x", padx=10, pady=10)
        header.pack_propagate(False)

        title = Label(header, text="🌳 Your EcoTree Growth", font=("Segoe UI", 20, "bold"),
                      bg="#2e7d32", fg="white")
        title.pack(pady=20)

        # Content Frame
        content = Frame(tree_window, bg="#f0f8f0")
        content.pack(fill="both", expand=True, padx=20, pady=10)

        # Create tree canvas
        tree_canvas = Canvas(content, width=600, height=500, bg="#e8f5e8", highlightthickness=0)
        tree_canvas.pack(fill="both", expand=True)

        # Draw tree based on current progress
        self.draw_detailed_tree(tree_canvas, self.items_recycled)

        # Progress info
        progress_frame = Frame(content, bg="#f0f8f0")
        progress_frame.pack(fill="x", pady=10)

        progress_text = f"🌱 Devices Recycled: {self.items_recycled} | Keep going to grow your tree!"
        progress_label = Label(progress_frame, text=progress_text, font=("Segoe UI", 12, "bold"),
                               bg="#f0f8f0", fg="#2e7d32")
        progress_label.pack()

        # Close button
        close_btn = Button(tree_window, text="Close Tree View",
                           command=tree_window.destroy,
                           bg="#4caf50", fg="white", font=("Segoe UI", 12, "bold"),
                           width=15, height=2)
        close_btn.pack(pady=10)

    def create_professional_pie_chart(self, parent, metals_dict):
        """Create professional pie chart in the given parent frame"""
        for widget in parent.winfo_children():
            widget.destroy()

        try:
            # Set professional style
            plt.style.use('seaborn-v0_8')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                           gridspec_kw={'width_ratios': [2, 1]})

            metal_colors = {
                'copper': '#FF6B35',
                'aluminum': '#4ECDC4',
                'gold': '#FFD166',
                'silver': '#E8E8E8',
                'steel': '#6A8EAE',
                'iron': '#5D5D5D',
                'lead': '#8E8E8E',
                'zinc': '#A1C349',
                'tin': '#D4B483',
                'nickel': '#D1D1D1',
                'plastic': '#FFA577',
                'glass': '#88CCFF',
                'other': '#DDA0DD'
            }

            colors = []
            labels = []
            sizes = []

            for metal, percentage in metals_dict.items():
                metal_lower = metal.lower()
                color_found = False
                for key_color, color_value in metal_colors.items():
                    if key_color in metal_lower:
                        colors.append(color_value)
                        color_found = True
                        break
                if not color_found:
                    colors.append(metal_colors['other'])
                labels.append(metal)
                sizes.append(percentage)

            explode = [0.05] * len(sizes)

            wedges, texts, autotexts = ax1.pie(sizes,
                                              labels=labels,
                                              autopct='%1.1f%%',
                                              startangle=90,
                                              colors=colors,
                                              explode=explode,
                                              shadow=True,
                                              textprops={'fontsize': 11, 'fontweight': 'bold'},
                                              wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'alpha': 0.9})

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)

            ax1.set_title("Material Composition Analysis\n(Recovery Value Shown Separately)",
                          fontsize=16, fontweight='bold', pad=25, color='#2e7d32')
            ax1.axis('equal')

            legend_elements = []
            for metal, color in zip(labels, colors):
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color,
                                                    label=f"{metal}", edgecolor='white', linewidth=1))

            ax2.axis('off')
            legend = ax2.legend(handles=legend_elements, loc='center', fontsize=11,
                                title="Materials Breakdown", title_fontsize=13,
                                frameon=True, fancybox=True, shadow=True,
                                framealpha=0.95, edgecolor='#2e7d32')
            legend.get_frame().set_facecolor('#f8fdf8')
            legend.get_title().set_color('#2e7d32')

            fig.patch.set_facecolor('#f8fdf8')

            plt.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        except Exception as e:
            print(f"Pie chart error: {e}")
            error_label = Label(parent,
                                text="Error displaying material composition\nPlease try scanning another device",
                                font=("Segoe UI", 12), bg="#f0f8f0", fg="red", justify=CENTER)
            error_label.pack(expand=True)

    def draw_detailed_tree(self, canvas, stage):
        """Draw detailed tree in tree window"""
        canvas.delete("all")

        # Background
        canvas.create_rectangle(0, 0, 600, 500, fill="#e8f5e8", outline="")
        canvas.create_rectangle(0, 0, 600, 300, fill="#87CEEB", outline="")
        canvas.create_rectangle(0, 300, 600, 500, fill="#8BC34A", outline="")

        canvas.create_oval(50, 50, 150, 100, fill="white", outline="white")
        canvas.create_oval(100, 40, 200, 90, fill="white", outline="white")
        canvas.create_oval(450, 70, 550, 120, fill="white", outline="white")

        tree_x, tree_base = 300, 450

        if stage == 0:
            canvas.create_oval(295, 430, 305, 440, fill="#5D4037", outline="#3E2723", width=2)
            canvas.create_text(300, 400, text="🌱 Plant your first device to grow your EcoTree!",
                               font=("Segoe UI", 14, "bold"), fill="#2e7d32")
        elif 1 <= stage <= 2:
            canvas.create_rectangle(298, 400, 302, 450, fill="#5D4037", outline="#3E2723")
            canvas.create_oval(270, 350, 330, 410, fill="#81C784", outline="#33691E")
            canvas.create_text(300, 320, text="🌿 Your EcoTree is sprouting!",
                               font=("Segoe UI", 14, "bold"), fill="#2e7d32")
        elif 3 <= stage <= 4:
            canvas.create_rectangle(295, 350, 305, 450, fill="#5D4037", outline="#3E2723")
            canvas.create_oval(240, 280, 360, 380, fill="#4CAF50", outline="#33691E")
            canvas.create_text(300, 250, text="🌳 Your tree is growing well!",
                               font=("Segoe UI", 14, "bold"), fill="#2e7d32")
        elif 5 <= stage <= 6:
            canvas.create_rectangle(292, 300, 308, 450, fill="#5D4037", outline="#3E2723")
            canvas.create_oval(200, 200, 400, 350, fill="#388E3C", outline="#33691E")
            canvas.create_oval(220, 250, 230, 260, fill="#FF5252", outline="#C62828")
            canvas.create_oval(380, 280, 390, 290, fill="#FF5252", outline="#C62828")
            canvas.create_text(300, 180, text="🌲 Your EcoTree is thriving!",
                               font=("Segoe UI", 14, "bold"), fill="#2e7d32")
        else:
            canvas.create_rectangle(290, 250, 310, 450, fill="#5D4037", outline="#3E2723")
            canvas.create_oval(150, 150, 450, 320, fill="#2E7D32", outline="#33691E")
            for x, y in [(180, 200), (220, 180), (380, 220), (420, 190), (300, 160)]:
                canvas.create_oval(x, y, x + 8, y + 8, fill="#FF5252", outline="#C62828")
            canvas.create_text(100, 120, text="🐦", font=("Arial", 20))
            canvas.create_text(500, 140, text="🐦", font=("Arial", 20))
            canvas.create_text(300, 130, text="🏆 Mature EcoTree! 🌟",
                               font=("Segoe UI", 16, "bold"), fill="#2e7d32")

    # -------------------- Frame Update --------------------
    def update_frame(self):
        if self.running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (640, 480))
                    frame = self.detect_device(frame)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.config(image=imgtk)
                else:
                    self.update_result_text("❌ Failed to read frame from camera")
                self.root.after(10, self.update_frame)
            except Exception as e:
                print(f"Frame update error: {e}")
                self.root.after(100, self.update_frame)
        else:
            self.root.after(1000, self.update_frame)

    # -------------------- Detection Logic --------------------
    def detect_device(self, frame):
        if model is None:
            cv2.putText(frame, "MODEL NOT LOADED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        if metal_data is None:
            cv2.putText(frame, "DATASET NOT LOADED", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame

        try:
            results = model(frame, verbose=False)
            boxes = results[0].boxes
            detected_label = None

            if boxes is not None and len(boxes) > 0:
                cls_id = int(boxes[0].cls[0])
                conf = float(boxes[0].conf[0])
                if conf >= 0.6:
                    detected_label = model.names[cls_id]
                    self.label_history.append(detected_label)
                    if self.detection_start_time is None:
                        self.detection_start_time = time.time()
                else:
                    self.label_history.clear()
                    self.detection_start_time = None
            else:
                self.label_history.clear()
                self.detection_start_time = None

            # Confirm device after 4 seconds
            if self.detection_start_time and (time.time() - self.detection_start_time >= 4):
                if self.label_history:
                    self.fixed_label = Counter(self.label_history).most_common(1)[0][0]
                    fixed_label_clean = self.fixed_label.strip().lower()

                    # Find match in dataset
                    matched = metal_data[metal_data['normalized_name'].str.lower() == fixed_label_clean]
                    if matched.empty:
                        closest_match = get_close_matches(fixed_label_clean, device_names_list, n=1, cutoff=0.5)
                        if closest_match:
                            matched = metal_data[metal_data['normalized_name'] == closest_match[0]]
                        else:
                            self.update_result_text(f"⚠ No match found for: {self.fixed_label}")
                            self.label_history.clear()
                            self.detection_start_time = None
                            self.fixed_label = None
                            return frame

                    if not matched.empty:
                        # Build pie_data ONLY from DESIRED_PIE_COLS_RAW (fuzzy match against actual columns)
                        row = matched.iloc[0]
                        actual_cols = list(row.index)

                        # prepare normalized map for actual columns
                        norm_actual = { _normalize_name(c): c for c in actual_cols }

                        pie_entries = []  # list of tuples: (desired_raw, actual_col, friendly_label, pct)
                        composition_text = f"🎯 Device Detected: {self.fixed_label}\n\n"
                        composition_text += "🔩 Material Composition:\n"
                        total_points = 0.0

                        for desired in DESIRED_PIE_COLS_RAW:
                            desired_norm = _normalize_name(desired)
                            col = None
                            if desired_norm in norm_actual:
                                col = norm_actual[desired_norm]
                            else:
                                choices = list(norm_actual.keys())
                                matches = get_close_matches(desired_norm, choices, n=1, cutoff=0.6)
                                if matches:
                                    col = norm_actual[matches[0]]
                                else:
                                    col = None

                            if col is None:
                                continue

                            val = row.get(col, 0)
                            pct = 0.0
                            try:
                                if pd.isna(val):
                                    pct = 0.0
                                elif isinstance(val, str):
                                    pct = float(val.replace('%', '').strip())
                                else:
                                    pct = float(val)
                            except Exception:
                                pct = 0.0

                            if pct <= 0:
                                continue

                            # friendly label - use original column name cleaned
                            friendly_label = col.replace('_', ' ').title()
                            pie_entries.append((desired, col, friendly_label, pct))

                        # Create pie_data dict from pie_entries
                        pie_data = {entry[2]: entry[3] for entry in pie_entries}

                        # --- Recovery value calculation updated to use Excel column if present ---
                        # Attempt to find an "estimated recovery" column in the actual columns (fuzzy)
                        recovery_col = None
                        for nkey, acol in norm_actual.items():
                            if 'estimated' in nkey and 'recover' in nkey:
                                recovery_col = acol
                                break
                        # If not found by that heuristic, try contains 'estimated' or 'recovery'
                        if recovery_col is None:
                            for nkey, acol in norm_actual.items():
                                if 'estimated' in nkey or 'recovery' in nkey:
                                    recovery_col = acol
                                    break

                        recovery_value_rounded = None
                        if recovery_col is not None:
                            # Parse the value from excel cell (strip non-numeric except dot)
                            raw_val = row.get(recovery_col, None)
                            try:
                                if pd.isna(raw_val) or raw_val is None:
                                    recovery_value_rounded = None
                                else:
                                    raw_str = str(raw_val)
                                    # extract number (may contain commas or currency symbol)
                                    num_str = re.sub(r'[^\d\.\-]', '', raw_str)
                                    if num_str == "" or num_str == "." or num_str == "-":
                                        recovery_value_rounded = None
                                    else:
                                        recovery_value_rounded = int(round(float(num_str)))
                            except Exception:
                                recovery_value_rounded = None

                        # Fallback behavior: if the Excel recovery value wasn't found or parseable,
                        # use previous heuristic (metal-specific sum or overall metal or sum of selected)
                        if recovery_value_rounded is None:
                            metal_specific_total = 0.0
                            overall_metal_value = None
                            for desired_raw, col, friendly_label, pct in pie_entries:
                                if _is_metal_specific(desired_raw):
                                    metal_specific_total += pct
                                if re.sub(r'[^a-z0-9]', '', desired_raw.lower()) == _normalize_name("Metal (%)"):
                                    overall_metal_value = pct

                            if metal_specific_total > 0:
                                recovery_value = metal_specific_total
                            elif overall_metal_value is not None:
                                recovery_value = overall_metal_value
                            else:
                                recovery_value = sum([entry[3] for entry in pie_entries])

                            recovery_value_rounded = int(round(recovery_value))

                        # Build composition text for display (only included friendly labels and pct)
                        for _, _, friendly_label, pct in pie_entries:
                            composition_text += f"  • {friendly_label}: {pct}%\n"
                        composition_text += f"\n💰 Estimated Recovery Value: ₹{recovery_value_rounded}"
                        composition_text += f"\n\n🌱 Eco Impact: You've recycled {self.items_recycled + 1} devices!"

                        # Update result text + quick stats and pie chart
                        self.update_result_text(composition_text)
                        self.update_quick_stats(self.fixed_label, recovery_value_rounded, pie_data)
                        self.current_pie_data = pie_data

                        # Update wallet & items (wallet stores rupee amount)
                        self.items_recycled += 1
                        self.wallet_balance += recovery_value_rounded
                        self.update_impact_display()

                        # Update small pie chart preview (main UI)
                        self.show_pie_chart(pie_data)

                    self.label_history.clear()
                    self.detection_start_time = None
                    self.fixed_label = None

            # Overlay text
            display_text = f"Detected: {detected_label}" if detected_label else "Scanning for devices..."
            cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print(f"Detection error: {e}")
            traceback.print_exc()
            cv2.putText(frame, f"Detection Error", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    # -------------------- Pie Chart --------------------
    def show_pie_chart(self, metals_dict):
        """Show the small pie chart in the impact panel's chart frame."""
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        if not metals_dict:
            # nothing to show
            placeholder = Label(self.chart_frame, text="No data yet", bg="#1b5e20", fg="white")
            placeholder.pack(expand=True)
            return
        fig, ax = plt.subplots(figsize=(3, 3), facecolor='#1b5e20')
        # choose colors automatically; make labels white for visibility
        wedges, texts, autotexts = ax.pie(list(metals_dict.values()), labels=list(metals_dict.keys()),
                                          autopct='%1.1f%%', startangle=90, textprops={'color': 'white', 'fontsize': 8})
        ax.set_title("Material Composition", color='white', fontsize=12)
        # embed
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        plt.close(fig)


# -------------------- Main --------------------
if __name__ == "__main__":
    print("Starting EcoTrace AI...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Data path: {DATA_PATH}")

    root = Tk()
    app = EcoTraceApp(root)

    def on_closing():
        app.running = False
        if app.cap:
            app.cap.release()
        cv2.destroyAllWindows()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
