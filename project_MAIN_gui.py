import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
import threading
from tkinter import filedialog, messagebox
from seam_carving_core import SeamCarverCore

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Smart Seam Carver - Content Aware Resizer")
        self.geometry("1150x700")
        self.core = None

        # --- طراحی Sidebar (منوی تنظیمات) ---
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        ctk.CTkLabel(self.sidebar, text="Seam Carving", font=("Helvetica", 24, "bold")).pack(pady=20)
        
        # دکمه بارگذاری: از متد ایمن برای مسیرهای فارسی استفاده می‌کند
        self.btn_load = ctk.CTkButton(self.sidebar, text="1. Load Image", command=self.load_image)
        self.btn_load.pack(pady=10)

        # دکمه پیش‌پردازش: اجرای کلاستربندی در ترد جداگانه
        self.btn_pre = ctk.CTkButton(self.sidebar, text="2. Run AI Pre-process", state="disabled", command=self.run_pre_thread)
        self.btn_pre.pack(pady=10)

        self.mode_var = ctk.StringVar(value="Vertical")
        self.mode_menu = ctk.CTkOptionMenu(self.sidebar, values=["Vertical", "Horizontal", "Smart"], variable=self.mode_var)
        self.mode_menu.pack(pady=10)

        self.entry_n = ctk.CTkEntry(self.sidebar, placeholder_text="Pixels to remove")
        self.entry_n.pack(pady=10)

        self.btn_run = ctk.CTkButton(self.sidebar, text="3. Start Processing", state="disabled", fg_color="#27ae60", command=self.run_main_thread)
        self.btn_run.pack(pady=10)

        self.progress = ctk.CTkProgressBar(self.sidebar)
        self.progress.set(0)
        self.progress.pack(pady=30, padx=20)

        # --- طراحی بدنه اصلی (نمایش تصاویر) ---
        self.display = ctk.CTkFrame(self)
        self.display.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        self.lbl_orig = ctk.CTkLabel(self.display, text="Original View")
        self.lbl_orig.pack(side="left", expand=True)

        self.lbl_res = ctk.CTkLabel(self.display, text="Processed Result")
        self.lbl_res.pack(side="right", expand=True)

    def load_image(self):
        """ خواندن تصویر با متد imdecode برای پشتیبانی از کاراکترهای فارسی در مسیر فایل """
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if path:
            try:
                # خواندن فایل به صورت باینری (حل مشکل NoneType)
                raw_data = np.fromfile(path, np.uint8)
                img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
                
                if img is not None:
                    self.core = SeamCarverCore(img)
                    self.update_display(img, is_orig=True)
                    self.update_display(img, is_orig=False)
                    self.btn_pre.configure(state="normal")
                else:
                    raise Exception("Decode Failed")
            except Exception as e:
                messagebox.showerror("Error", f"Could not read the file! \nCheck path for special characters.")

    def update_display(self, img, is_orig=False):
        """ تبدیل ماتریس OpenCV به فرمت قابل نمایش در رابط کاربری (PIL Image) """
        img_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((480, 480)) # تغییر اندازه برای نمایش متناسب در پنل
        ctk_img = ctk.CTkImage(img_pil, size=img_pil.size)
        if is_orig: self.lbl_orig.configure(image=ctk_img, text="")
        else: self.lbl_res.configure(image=ctk_img, text="")

    def run_pre_thread(self):
        """ اجرای مرحله پیش‌پردازش در یک ترد جداگانه برای جلوگیری از فریز شدن UI """
        self.btn_pre.configure(state="disabled")
        self.progress.configure(mode="indeterminate")
        self.progress.start()
        threading.Thread(target=self.do_pre, daemon=True).start()

    def do_pre(self):
        """ وظیفه‌ای که در پس‌زمینه اجرا می‌شود (K-Means) """
        self.core.preprocess_step(k=64)
        self.after(0, self.finish_pre)

    def finish_pre(self):
        """ بازگشت به ترد اصلی بعد از اتمام پیش‌پردازش """
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(1.0)
        self.update_display(self.core.current_image, False)
        self.btn_run.configure(state="normal")
        messagebox.showinfo("Success", "AI Analysis & Quantization Completed!")

    def run_main_thread(self):
        """ شروع فرآیند اصلی Seam Carving با مدیریت ترد """
        try:
            n = int(self.entry_n.get())
            mode = self.mode_var.get()
            valid, limit = self.core.validate_request(n, mode)
            if not valid:
                messagebox.showwarning("Limit Warning", f"Requested size is too small. Max allowed: {limit}")
                return
            
            self.btn_run.configure(state="disabled")
            threading.Thread(target=self.do_resize, args=(n, mode), daemon=True).start()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid integer for pixel count.")

    def do_resize(self, n, mode):
        """ حلقه اصلی حذف درزها با آپدیت لحظه‌ای نوار پیشرفت """
        for i in range(n):
            if mode == "Vertical": 
                self.core.remove_vertical_seam()
            elif mode == "Horizontal":
                self.core.current_image = np.rot90(self.core.current_image, 1)
                self.core.remove_vertical_seam()
                self.core.current_image = np.rot90(self.core.current_image, 3)
            elif mode == "Smart":
                # تصمیم‌گیری هوشمند: حذف جهتی که کمترین انرژی را مصرف می‌کند
                if self.core.get_seam_energy('v') <= self.core.get_seam_energy('h'):
                    self.core.remove_vertical_seam()
                else:
                    self.core.current_image = np.rot90(self.core.current_image, 1)
                    self.core.remove_vertical_seam()
                    self.core.current_image = np.rot90(self.core.current_image, 3)
            
            # آپدیت نوار پیشرفت و تصویر در هر مرحله (استفاده از after برای ایمنی ترد)
            self.after(0, lambda v=(i+1)/n: self.progress.set(v))
            if i % 10 == 0: # برای افزایش سرعت، تصویر هر 10 مرحله یکبار آپدیت می‌شود
                self.after(0, lambda: self.update_display(self.core.current_image, False))
        
        self.after(0, lambda: self.update_display(self.core.current_image, False))
        self.after(0, lambda: messagebox.showinfo("Finished", "Smart Resizing completed successfully!"))
        self.after(0, lambda: self.btn_run.configure(state="normal"))

if __name__ == "__main__":
    app = App()
    app.mainloop()