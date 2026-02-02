import cv2
import numpy as np

class SeamCarverCore:
    """
    کلاس هسته‌ی الگوریتم Seam Carving
    
    ویژگی‌ها:
    - محاسبات برداری (Vectorized) برای سرعت بالا
    - حالت Smart برای مقایسه انرژی افقی و عمودی
    - متد Static برای قابلیت Multi-scale
    """

    def __init__(self, image):
        """
        سازنده کلاس
        image: تصویر ورودی (BGR)
        """
        # تبدیل به float64 برای جلوگیری از خطای سرریز در محاسبات Sobel
        self.image = image.astype(np.float64)

    def energy_map(self, img):
        """ محاسبه نقشه انرژی با LAB و Sobel """
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
        L, _, _ = cv2.split(lab)
        L = cv2.GaussianBlur(L, (3, 3), 0)

        gx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
        return np.abs(gx) + np.abs(gy)

    def find_vertical_seam(self, energy):
        """ یافتن درز با DP (نسخه سریع Vectorized) """
        h, w = energy.shape
        cost = energy.copy()
        
        # محاسبه ماتریس هزینه تجمعی
        for i in range(1, h):
            m1 = cost[i-1, :-2]
            m2 = cost[i-1, 1:-1]
            m3 = cost[i-1, 2:]
            cost[i, 1:-1] += np.minimum(m1, np.minimum(m2, m3))
            cost[i, 0] += min(cost[i-1, 0], cost[i-1, 1])
            cost[i, -1] += min(cost[i-1, -1], cost[i-1, -2])

        # بازگشت (Backtracking)
        seam = np.zeros(h, dtype=int)
        seam[-1] = np.argmin(cost[-1])
        total_energy = cost[-1, seam[-1]]

        for i in range(h-2, -1, -1):
            prev_x = seam[i+1]
            start, end = max(0, prev_x-1), min(w, prev_x+2)
            seam[i] = start + np.argmin(cost[i, start:end])
        
        return seam, total_energy

    def remove_vertical_seam(self, img, seam):
        """ حذف فیزیکی درز """
        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=bool)
        mask[np.arange(h), seam] = False
        return img[mask].reshape(h, w - 1, 3)

    def step(self, mode="smart"):
        """ اجرای یک گام (هوشمند یا معمولی) """
        img = self.image
        actual_mode = mode

        if mode == "smart":
            ev = self.energy_map(img)
            _, cost_v = self.find_vertical_seam(ev)
            
            img_h = np.rot90(img, 1, (0, 1))
            eh = self.energy_map(img_h)
            _, cost_h = self.find_vertical_seam(eh)
            
            actual_mode = "vertical" if cost_v <= cost_h else "horizontal"

        work_img = np.rot90(self.image, 1, (0, 1)) if actual_mode == "horizontal" else self.image
        energy = self.energy_map(work_img)
        seam_idx, _ = self.find_vertical_seam(energy)

        preview = work_img.copy()
        preview[np.arange(work_img.shape[0]), seam_idx] = [0, 0, 255]

        res_img = self.remove_vertical_seam(work_img, seam_idx)

        if actual_mode == "horizontal":
            res_img = np.rot90(res_img, 3, (0, 1))
            preview = np.rot90(preview, 3, (0, 1))

        self.image = res_img
        return preview.astype(np.uint8), res_img.astype(np.uint8)

    @staticmethod
    def downscale(img, scale=0.5):
        """ 
        بخش Multi-scale: 
        کوچک کردن تصویر قبل از پردازش برای افزایش سرعت در تصاویر بزرگ
        """
        h, w = img.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
