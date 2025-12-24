import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class SeamCarverCore:
    def __init__(self, image):
        """
        مقداردهی اولیه کلاس با تصویر ورودی
        تصویر به float64 تبدیل می‌شود تا در محاسبات فیلترها دقت حفظ شود
        """
        self.original_image = image.astype(np.float64)
        self.current_image = np.copy(self.original_image)

    def preprocess_step(self, k=64):
        """
        مرحله پیش‌پردازش هوشمند:
        1. انتقال به فضای LAB برای جدا کردن روشنایی از رنگ
        2. استفاده از هوش مصنوعی (K-Means) برای ساده‌سازی بافت تصویر
        """
        # تبدیل فضای رنگی: کانال L برای ما حیاتی است چون لبه‌ها در آن واضح‌ترند
        lab = cv2.cvtColor(self.current_image.astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # --- استراتژی سرعت (Statistical Sampling) ---
        # برای یادگیری رنگ‌ها، نیازی به تمام پیکسل‌ها نیست. یک نسخه کوچک می‌سازیم
        h, w = l.shape
        scale = 300.0 / max(h, w)
        l_small = cv2.resize(l, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        pixels_small = l_small.reshape((-1, 1))

        # تنظیمات بهینه K-Means برای جلوگیری از لوپ و افزایش سرعت
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=1024,
            max_iter=10,           # محدود کردن تکرار برای جلوگیری از گیر کردن
            n_init=1,
            max_no_improvement=3,  # خروج سریع در صورت عدم بهبود
            random_state=42
        )
        
        # مرحله آموزش (Fit) فقط روی نمونه کوچک انجام می‌شود
        kmeans.fit(pixels_small)
        
        # مرحله اعمال (Predict) روی تمام پیکسل‌های اصلی (رزولوشن کامل) انجام می‌شود
        pixels_full = l.reshape((-1, 1))
        labels_full = kmeans.predict(pixels_full)
        centers = kmeans.cluster_centers_.astype(np.uint8)
        quantized_l = centers[labels_full].reshape(l.shape)
        
        # بازسازی تصویر: ترکیب کانال L تغییر یافته با کانال‌های رنگی اصلی (a, b)
        self.current_image = cv2.cvtColor(cv2.merge([quantized_l, a, b]), cv2.COLOR_LAB2BGR).astype(np.float64)
        return self.current_image.astype(np.uint8)

    def compute_energy(self):
        """
        محاسبه اهمیت هر پیکسل (Energy Map)
        استفاده از لبه‌یاب لاپلاسین (Laplacian) پس از نرم کردن تصویر با فیلتر گاوسی
        """
        gray = cv2.cvtColor(self.current_image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # حذف نویزهای فرکانس بالا برای جلوگیری از ایجاد لبه‌های کاذب
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # مشتق دوم تصویر: هر چه مقدار بزرگتر باشد، آن نقطه لبه مهم‌تری است
        return np.abs(cv2.Laplacian(blurred, cv2.CV_64F))

    def get_seam_energy(self, mode='v'):
        """
        تخمین انرژی کل یک درز برای تصمیم‌گیری در حالت Smart Mode
        این تابع به برنامه اجازه می‌دهد بین حذف افقی و عمودی، بهینه‌ترین را انتخاب کند
        """
        temp_img = np.rot90(self.current_image, 1) if mode == 'h' else self.current_image
        gray = cv2.cvtColor(temp_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        energy = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_64F))
        
        cost = energy.copy()
        for i in range(1, energy.shape[0]):
            cost[i, 1:-1] += np.minimum(cost[i-1, :-2], np.minimum(cost[i-1, 1:-1], cost[i-1, 2:]))
        return np.min(cost[-1, :])

    def remove_vertical_seam(self):
        """
        پیاده‌سازی اصلی Seam Carving با برنامه‌نویسی پویا (DP)
        یافتن کم‌انرژی‌ترین مسیر از بالا به پایین و حذف فیزیکی آن
        """
        energy = self.compute_energy()
        rows, cols = energy.shape
        cost = energy.copy()
        
        # گام اول: محاسبه ماتریس هزینه تجمعی (Forward Pass)
        for i in range(1, rows):
            # برای هر پیکسل، مینیمم هزینه مسیرهای بالایی را پیدا می‌کنیم
            cost[i, 1:-1] += np.minimum(cost[i-1, :-2], np.minimum(cost[i-1, 1:-1], cost[i-1, 2:]))
            cost[i, 0] += min(cost[i-1, 0], cost[i-1, 1])
            cost[i, -1] += min(cost[i-1, -1], cost[i-1, -2])

        # گام دوم: ردیابی مسیر بهینه از پایین به بالا (Backtracking)
        seam = np.zeros(rows, dtype=int)
        seam[-1] = np.argmin(cost[-1, :])
        for i in range(rows - 2, -1, -1):
            s = max(0, seam[i+1]-1)
            e = min(cols, seam[i+1]+2)
            seam[i] = s + np.argmin(cost[i, s:e])

        # گام سوم: حذف مسیر یافت شده با استفاده از ماسک منطقی در NumPy
        mask = np.ones((rows, cols), dtype=bool)
        mask[np.arange(rows), seam] = False
        # بازسازی ماتریس تصویر با یک ستون کمتر
        self.current_image = self.current_image[mask].reshape(rows, cols - 1, 3)

    def validate_request(self, n, mode):
        """ بررسی محدودیت‌های مجاز برای تغییر سایز (جلوگیری از نابودی کامل تصویر) """
        r, c = self.current_image.shape[:2]
        limit = int(min(r, c) * 0.6) # محدودیت 60 درصدی برای حفظ کیفیت
        return n <= limit, limit