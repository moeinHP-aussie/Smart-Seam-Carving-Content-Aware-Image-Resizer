<div align="center">
  <h1> " Smart Seam Carving: Content-Aware Image Resizer"</h1>
  <p><b>A high-performance seam carving tool with multi-scale support and intelligent seam selection</b></p>
</div>

<hr>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Algorithm-Dynamic%20Programming-red.svg" alt="Algorithm">
  <img src="https://img.shields.io/badge/Domain-Image%20Processing-cyan.svg" alt="Domain">
  <img src="https://img.shields.io/badge/UI-PyQt6-orange.svg" alt="UI Framework">
  <img src="https://img.shields.io/badge/Optimization-Vectorized%20NumPy-green.svg" alt="Optimization">
  <img src="https://img.shields.io/badge/Feature-Multi--scale-yellow.svg" alt="Multi-scale">
  <img src="https://img.shields.io/badge/Boundary-Infinity%20Padding-brightgreen.svg" alt="Boundary Handling">
</p>

<h2>📌 Overview</h2>
<p align="justify">
This project implements a <b>Content-Aware Image Resizing</b> tool based on the Seam Carving algorithm. Unlike traditional scaling which distorts the image or cropping which removes edges, this tool intelligently removes "seams" (paths of least importance) to resize images while preserving essential visual content. The implementation features a modern PyQt6 GUI, multi-scale processing for speed, and a smart mode that automatically selects the optimal seam direction with proper boundary condition handling.
</p>


## 🖼️ Results

The following examples demonstrate the effectiveness of the smart seam carving algorithm on different types of images.

| Example | Original | Content-Aware Result |
|:-------:|:--------:|:--------------------:|
| **Cats** | <img src="sample%20pictures/cats.jpg" width="280"/> | <img src="sample%20pictures/cats_resize_smart.png" width="280"/> |
| **Wildlife** | <img src="sample%20pictures/wildlife.jpg" width="280"/> | <img src="sample%20pictures/wildlife_resize_smart.png" width="280"/> |
| **Toy Story**<br>(Horizontal Resize) | <img src="sample%20pictures/Toy%20story.jpg" width="280"/> | <img src="sample%20pictures/toystory_resize_horizontal.png" width="280"/> |
| **Soul**<br>(Vertical Resize) | <img src="sample%20pictures/soul.jpg" width="280"/> | <img src="sample%20pictures/soul_resize_vertical.png" width="280"/> |
| **Adam and Eve**<br>(Vertical Resize) | <img src="sample%20pictures/Creating-Adam-And-Eve-By-Kevin-Wood-2.jpeg" width="280"/> | <img src="sample%20pictures/Creating-Adam-And-Eve-By-Kevin-Wood-2_resize_vertical.png" width="280"/> |
| **Sports Photography**<br>(Horizontal Resize) | <img src="sample%20pictures/winners-world-sports-photography-awards-2023-1.jpeg" width="280"/> | <img src="sample%20pictures/winners-world-sports-photography-awards-2023-1_resize_horizontal.jpg" width="280"/> |

---

## 🖥️ Graphical User Interface

The application provides an intuitive GUI for loading images, selecting resizing modes, and visualizing the output.

<p align="center">
    <img src="sample%20pictures/Screenshot%202026-07-02%20210910.png" width="900">
</p>


<h2>✨ Key Features</h2>
<ul>
  <li><b>Smart Mode:</b> Automatically chooses between horizontal and vertical seam removal based on global energy cost comparison</li>
  <li><b>Multi-scale Support:</b> Optional downscaling before processing to dramatically speed up operations on large images</li>
  <li><b>Optimized Vectorized DP:</b> Fully vectorized Dynamic Programming implementation with infinity padding for boundary conditions</li>
  <li><b>Modern PyQt6 GUI:</b> Responsive, multi-threaded interface with live preview and progress tracking</li>
  <li><b>Boundary-Aware Algorithm:</b> Proper handling of edge pixels using infinity padding technique</li>
  <li><b>Real-time Preview:</b> Visualizes seams in red during processing for better user insight</li>
  <li><b>High Performance:</b> Vectorized NumPy operations for near-C speed</li>
</ul>

<h2>🛠 Technical Implementation</h2>

<h3>1️⃣ Energy Map Calculation</h3>
<p>
The image is converted to the <b>LAB color space</b>, and the <b>L-channel (Lightness)</b> is extracted. Gaussian blur is applied to reduce noise, followed by <b>Sobel operators</b> to compute gradients. The energy map is defined as the sum of absolute gradients:
</p>
<p align="center">
  $E = |G_x| + |G_y|$
</p>

```python
def energy_map(self, img):
    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)
    L = cv2.GaussianBlur(L, (3, 3), 0)

    gx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    return np.abs(gx) + np.abs(gy) 
```
<h3>2️⃣ Boundary-Aware Seam Finding with Infinity Padding</h3> <p> The core innovation in this implementation is the use of <b>infinity padding</b> for handling boundary conditions in the DP algorithm. This elegant approach eliminates conditional statements and allows for fully vectorized operations: </p>

```python

def find_vertical_seam(self, energy):


    h, w = energy.shape
    cost = energy.copy()
    
    for i in range(1, h):
        prev_row = cost[i-1]
        left = np.insert(prev_row[:-1], 0, np.inf)    # Pad left with infinity
        right = np.append(prev_row[1:], np.inf)       # Pad right with infinity
        center = prev_row
        
        cost[i] += np.minimum(center, np.minimum(left, right))

    # Backtracking برای پیدا کردن مسیر درز
    seam = np.zeros(h, dtype=int)
    seam[-1] = np.argmin(cost[-1])

    for i in range(h-2, -1, -1):
        prev_x = seam[i+1]
        start = max(0, prev_x - 1)
        end = min(w, prev_x + 2)
        seam[i] = start + np.argmin(cost[i, start:end])
    
    return seam, cost[-1, seam[-1]]

 ```
<h3>3️⃣ Multi-scale Processing Pipeline</h3> <p> For large images, the algorithm can optionally downsample before processing to significantly improve performance: </p>

```python
  @staticmethod
  def downscale(img, scale=0.5):
      h, w = img.shape[:2]
      return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
```
<h3>4️⃣ Intelligent Direction Selection (Smart Mode)</h3> <p> In <b>Smart Mode</b>, the algorithm computes the total energy cost for both vertical and horizontal seams and chooses the direction with the lower cost: </p>
```python

def step(self, mode="smart"):

    img = self.image
    actual_mode = mode

    if mode == "smart":
        ev = self.energy_map(img)
        _, cost_v = self.find_vertical_seam(ev)
        
        img_h = np.rot90(img, 1, (0, 1))
        eh = self.energy_map(img_h)
        _, cost_h = self.find_vertical_seam(eh)
        
        actual_mode = "vertical" if cost_v <= cost_h else "horizontal"
    
    # ...
```

<h3>5️⃣ Seam Removal and Image Reconstruction</h3> <p> After finding the optimal seam, it's removed from the image using boolean masking: </p>

```python
def remove_vertical_seam(self, img, seam):
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False
    return img[mask].reshape(h, w - 1, 3)
```

<h2>📁 Project Structure</h2>

<h3>Visual Features:</h3> <ul> <li>Seams are shown in red during processing</li> <li>Real-time updates during multi-threaded processing</li> <li>Responsive layout that adapts to window size</li> <li>Professional styling with green "Run" button</li> </ul><h2>⚡ Performance Optimization</h2><h3>Multi-scale Processing:</h3> <p> When the "Multi-scale" option is enabled, the image is downscaled to 50% of its original size before processing. This can reduce processing time by up to <b>75%</b> on high-resolution images while maintaining visual quality. </p><h3>Vectorized Operations:</h3> <p> All critical computations use NumPy's vectorized operations instead of Python loops: </p> <ul> <li>Energy map calculation with OpenCV filters</li> <li>DP matrix computation with np.minimum</li> <li>Seam removal with boolean array masking</li> </ul><h3>Memory Efficiency:</h3> <ul> <li>Images processed in float64 precision to prevent overflow</li> <li>In-place operations where possible</li> <li>Efficient array slicing instead of copying</li> </ul><h2>🔬 Technical Details</h2><h3>Algorithm Complexity:</h3> <ul> <li><b>Time Complexity:</b> O(n × m) where n is number of seams and m is image pixels</li> <li><b>Space Complexity:</b> O(w × h) for the DP cost matrix</li> <li><b>Optimized for:</b> Large images with multi-scale, small images with full resolution</li> </ul><h3>Color Space Selection:</h3> <p> The algorithm uses <b>LAB color space</b> instead of RGB because: </p> <ul> <li>LAB separates luminance (L) from color channels</li> <li>Human vision is more sensitive to luminance changes</li> <li>Better edge detection in the L-channel</li> </ul><h3>Edge Detection Methodology:</h3> <p> Sobel operators are preferred over other edge detectors because: </p> <ul> <li>Computationally efficient</li> <li>Provides directional gradient information</li> <li>Less sensitive to noise than simple gradient methods</li> <li>Well-suited for energy map creation</li> </ul><h2>🎯 Use Cases</h2><h3>Ideal Applications:</h3> <ul> <li><b>Content-Aware Image Resizing:</b> Resize images without distorting important content</li> <li><b>Image Retargeting:</b> Adapt images to different aspect ratios</li> <li><b>Object Removal:</b> Remove unwanted objects by repeatedly removing seams</li> <li><b>Educational Tool:</b> Learn about dynamic programming and image processing</li> </ul><h3>Example Scenarios:</h3> <ol> <li><b>Web Design:</b> Adapt product images to fit different container sizes</li> <li><b>Mobile Development:</b> Create responsive images for various screen sizes</li> <li><b>Photography:</b> Adjust image composition without cropping</li> <li><b>Research:</b> Experiment with content-aware image manipulation</li> </ol><h2>⚠️ Limitations & Considerations</h2><h3>Current Limitations:</h3> <ul> <li>Only supports seam removal (not insertion)</li> <li>Batch processing not implemented</li> <li>No protection masks for important regions</li> <li>Limited to RGB/BGR color images</li> </ul><h3>Best Practices:</h3> <ol> <li>Use Multi-scale for images larger than 2MP for better performance</li> <li>Start with small seam counts and gradually increase</li> <li>Use Smart mode for general-purpose resizing</li> <li>Save original images before extensive processing</li> </ol><h2>🔄 Comparison with Traditional Methods</h2><table> <tr> <th>Method</th> <th>Advantages</th> <th>Disadvantages</th> </tr> <tr> <td><b>Traditional Scaling</b></td> <td>Simple, fast</td> <td>Distorts entire image uniformly</td> </tr> <tr> <td><b>Cropping</b></td> <td>Preserves quality in selected region</td> <td>Loses content at edges</td> </tr> <tr> <td><b>Seam Carving (This Project)</b></td> <td>Preserves important content, removes low-energy regions</td> <td>Computationally intensive, can create artifacts in complex scenes</td> </tr> </table><h2>🚀 Future Improvements</h2><h3>Planned Features:</h3> <ul> <li>Seam insertion for image enlargement</li> <li>Protection masks for important regions</li> <li>Batch processing mode</li> <li>GPU acceleration support</li> <li>Additional energy functions</li> <li>Video seam carving support</li> </ul><h3>Research Directions:</h3> <ul> <li>Deep learning-based energy functions</li> <li>Real-time seam carving for video</li> <li>3D seam carving for volumetric data</li> <li>Multi-objective seam optimization</li> </ul><h2>🤝 Contributing</h2><p>Contributions are welcome! Here's how you can help:</p><ol> <li><b>Report Bugs:</b> Open an issue with detailed bug reports</li> <li><b>Suggest Features:</b> Propose new features or improvements</li> <li><b>Submit Code:</b> Create pull requests with well-documented changes</li> <li><b>Improve Documentation:</b> Help enhance this README or add tutorials</li> </ol>
project/
├── project_MAIN_gui.py      # Main PyQt6 GUI application
├── seam_carving_core.py     # Core seam carving algorithm
└── README.md                # This documentation

<h2>📦 Installation & Requirements</h2><pre><code>pip install numpy opencv-python PyQt6</code></pre><h3>Dependencies:</h3> <ul> <li><b>Python 3.10+</b>: Required for modern Python features</li> <li><b>NumPy</b>: For efficient numerical computations</li> <li><b>OpenCV-Python</b>: For image processing operations</li> <li><b>PyQt6</b>: For the graphical user interface</li> </ul>

