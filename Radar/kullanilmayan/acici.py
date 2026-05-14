import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 0'dan 10'a kadar doğru üzerinde noktalar
t = np.linspace(0, 10, 100)

# x=y=z olacak şekilde doğru
x = t
y = t
z = t

# Vektör
v = np.array([1, 1, 1]) / np.sqrt(3)

# Eksen vektörleri
x_axis = np.array([1, 0, 0])
y_axis = np.array([0, 1, 0])
z_axis = np.array([0, 0, 1])

# Açı hesaplama (cos θ = a·b / |a||b|)
def angle_between(a, b):
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(cos_theta))

angle_x = angle_between(v, x_axis)
angle_y = angle_between(v, y_axis)
angle_z = angle_between(v, z_axis)

# 3D figür oluştur
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Doğruyu çiz
ax.plot(x, y, z, label="x=y=z doğrusu")

# Eksenlere vektörler çiz
ax.quiver(0,0,0, 1,0,0, color="r", length=5, normalize=True)
ax.quiver(0,0,0, 0,1,0, color="g", length=5, normalize=True)
ax.quiver(0,0,0, 0,0,1, color="b", length=5, normalize=True)
ax.quiver(0,0,0, v[0], v[1], v[2], color="k", length=5, normalize=True)

# Açıları yazdır
ax.text(3,0,0, f"{angle_x:.1f}°", color="r")
ax.text(0,3,0, f"{angle_y:.1f}°", color="g")
ax.text(0,0,3, f"{angle_z:.1f}°", color="b")

# Eksen isimleri
ax.set_xlabel("X ekseni")
ax.set_ylabel("Y ekseni")
ax.set_zlabel("Z ekseni")
ax.legend()

plt.show()
