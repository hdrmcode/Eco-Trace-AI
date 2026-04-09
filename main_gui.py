import cv2
import pandas as pd
import time
from collections import Counter
from difflib import get_close_matches
from ultralytics import YOLO
from tkinter import *
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------- Load YOLO model --------------------
MODEL_PATH = r"D:\Sprinthathon25\Main2\runs\retrain_from_last_40epochs\weights\best.pt"
model = YOLO(MODEL_PATH)

# -------------------- Load dataset --------------------
DATA_PATH = r"D:\Sprinthathon25\Main2\Filtered_Device_List.xlsx"
metal_data = pd.read_excel(DATA_PATH)

# Clean dataset columns
metal_data.columns = (metal_data.columns
                      .str.replace('\xa0', '', regex=False)
                      .str.strip()
                      .str.replace(' ', '_')
                      .str.lower())

if 'normalized_name' not in metal_data.columns:
    raise Exception("⚠ 'normalized_name' column not found!")

device_names_list = metal_data['normalized_name'].tolist()

# -------------------- Detection globals --------------------
label_history = []
detection_start_time = None
fixed_label = None

# -------------------- Tkinter App --------------------
class EcoTraceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 EcoTrace AI - Device Detection")
        self.root.geometry("1400x900")
        self.root.configure(bg="#e8f0f2")

        # -------------------- Camera --------------------
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("⚠ Could not open camera")
            exit()
        self.running = True

        # -------------------- Stats --------------------
        self.items_recycled = 0
        self.wallet_balance = 0

        # -------------------- UI Layout --------------------
        self.title_label = Label(root, text="🌿 EcoTrace AI", font=("Segoe UI", 26, "bold"), bg="#e8f0f2", fg="#2e7d32")
        self.title_label.pack(pady=10)

        # Video feed
        self.video_label = Label(root, bg="#cfd8dc", width=640, height=480)
        self.video_label.pack(pady=10)

        # Control Buttons
        control_frame = Frame(root, bg="#e8f0f2")
        control_frame.pack(pady=10)
        self.stop_btn = Button(control_frame, text="Stop Camera", command=self.stop_camera,
                               bg="#e53935", fg="white", font=("Segoe UI", 12, "bold"), width=15)
        self.stop_btn.grid(row=0, column=0, padx=10)

        # Results Display
        self.result_label = Label(root, text="Detection Results will appear here...", font=("Segoe UI", 12),
                                  bg="#ffffff", justify=LEFT, anchor="nw", width=60, height=12,
                                  relief="solid", bd=1, padx=10, pady=10)
        self.result_label.pack(pady=10)

        # Frame for pie chart and tree
        info_frame = Frame(root, bg="#e8f0f2")
        info_frame.pack(pady=10)

        # Pie chart area
        self.chart_frame = Frame(info_frame, bg="#e8f0f2")
        self.chart_frame.grid(row=0, column=0, padx=40)

        # Tree canvas
        self.tree_canvas = Canvas(info_frame, width=400, height=400, bg="#dcedc8",
                                  highlightthickness=2, highlightbackground="#558b2f")
        self.tree_canvas.grid(row=0, column=1)
        self.tree_canvas.create_text(200, 20, text="Your EcoTree 🌳", font=("Segoe UI", 14, "bold"), fill="#33691e")

        # Stats at bottom
        self.stats_label = Label(root, text="Items Recycled: 0 | Wallet Balance: ₹0",
                                 font=("Segoe UI", 12, "bold"), bg="#e8f0f2", fg="#37474f")
        self.stats_label.pack(pady=10)

        # -------------------- Detection variables --------------------
        self.label_history = []
        self.detection_start_time = None
        self.fixed_label = None

        # Draw initial tree
        self.draw_tree(0)

        # Start updating frames
        self.update_frame()

    # -------------------- Camera --------------------
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')

    # -------------------- Frame Update --------------------
    def update_frame(self):
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                frame = self.detect_device(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            self.video_label.after(10, self.update_frame)

    # -------------------- Detection Logic --------------------
    def detect_device(self, frame):
        results = model(frame, verbose=False)
        boxes = results[0].boxes
        detected_label = None

        if len(boxes) > 0:
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
            self.fixed_label = Counter(self.label_history).most_common(1)[0][0]
            fixed_label_clean = self.fixed_label.strip().lower()

            matched = metal_data[metal_data['normalized_name'].str.lower() == fixed_label_clean]
            if matched.empty:
                closest_match = get_close_matches(fixed_label_clean, device_names_list, n=1, cutoff=0.5)
                if closest_match:
                    matched = metal_data[metal_data['normalized_name'] == closest_match[0]]
                else:
                    self.result_label.config(text="⚠ No similar device found in dataset.")
                    self.label_history.clear()
                    self.detection_start_time = None
                    self.fixed_label = None
                    return frame

            metals = matched.iloc[0].to_dict()
            total_points = 0
            composition_text = f"✅ Device: {self.fixed_label}\n\n🔩 Metal Composition:\n"
            pie_data = {}

            for metal, percentage in metals.items():
                if metal != "normalized_name":
                    try:
                        percentage_value = float(str(percentage).replace('%', '').strip())
                    except ValueError:
                        percentage_value = 0
                    composition_text += f"  {metal}: {percentage_value}%\n"
                    pie_data[metal] = percentage_value
                    total_points += percentage_value

            composition_text += f"\n💰 Total Points: {total_points} points (₹{total_points})"
            self.result_label.config(text=composition_text)
            self.show_pie_chart(pie_data)

            # Update stats
            self.items_recycled += 1
            self.wallet_balance += total_points
            self.stats_label.config(text=f"Items Recycled: {self.items_recycled} | Wallet Balance: ₹{self.wallet_balance}")

            # Update tree
            self.draw_tree(self.items_recycled)

            self.label_history.clear()
            self.detection_start_time = None
            self.fixed_label = None

        # Overlay text
        display_text = f"Detected: {detected_label}" if detected_label else "Detecting..."
        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    # -------------------- Pie Chart --------------------
    def show_pie_chart(self, metals_dict):
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(metals_dict.values(), labels=metals_dict.keys(), autopct='%1.1f%%')
        ax.set_title("Material Composition")
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()
        plt.close(fig)

    # -------------------- Tree Growth --------------------
    def draw_tree(self, stage):
        self.tree_canvas.delete("tree")
        self.tree_canvas.create_rectangle(195, 350, 205, 400, fill="#5d4037", tags="tree")

        if stage == 0:
            self.tree_canvas.create_oval(190, 330, 210, 350, fill="#8bc34a", tags="tree")
        elif 1 <= stage <= 3:
            self.tree_canvas.create_oval(180, 300, 220, 340, fill="#66bb6a", tags="tree")
            self.tree_canvas.create_line(200, 350, 180, 310, width=3, fill="#558b2f", tags="tree")
        elif 4 <= stage <= 6:
            self.tree_canvas.create_oval(160, 260, 240, 320, fill="#43a047", tags="tree")
            self.tree_canvas.create_line(200, 350, 160, 280, width=4, fill="#33691e", tags="tree")
            self.tree_canvas.create_line(200, 350, 240, 280, width=4, fill="#33691e", tags="tree")
        elif stage >= 7:
            self.tree_canvas.create_oval(140, 220, 260, 300, fill="#2e7d32", tags="tree")
            self.tree_canvas.create_oval(170, 250, 230, 320, fill="#388e3c", tags="tree")
            self.tree_canvas.create_line(200, 350, 150, 250, width=5, fill="#1b5e20", tags="tree")
            self.tree_canvas.create_line(200, 350, 250, 250, width=5, fill="#1b5e20", tags="tree")

# -------------------- Main --------------------
if __name__ == "__main__":
    root = Tk()
    app = EcoTraceApp(root)
    root.mainloop()
