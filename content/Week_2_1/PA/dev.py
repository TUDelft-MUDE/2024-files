import sys
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3,
                    threshold=sys.maxsize,
                    floatmode='fixed')
plot = True

base = 2
height = np.sqrt(base**2 - (base/2)**2)

print(f'The base of the triangle is {base} units long '
      + f'and the height is {height} units long')

coord_x_1 = np.arange(1, 8, base)
coord_x_2 = np.arange(2, 7, base)

triangle_vertices = np.column_stack((coord_x_1, np.ones_like(coord_x_1)))

y_value = 1
for i in range(2, 4):
    y_value += height
    if i % 2 == 0:
        next_coord_x = coord_x_2
    else:
        next_coord_x = coord_x_1
    next_step = np.column_stack((next_coord_x,
                                 np.ones_like(next_coord_x)*y_value))
    triangle_vertices = np.vstack((triangle_vertices, next_step))
    

assert (
    np.isclose((base/2)**2 + height**2, 4)
), (
    'Distance is not correct'
)

print(triangle_vertices)

print(f'The dimensions of the vertices is {triangle_vertices.shape}')

scrambled = np.vstack((triangle_vertices[3,:],
                       triangle_vertices[5,:],
                       triangle_vertices[1,:],
                       triangle_vertices[2,:],
                       triangle_vertices[8,:],
                       triangle_vertices[7,:],
                       triangle_vertices[6,:],
                       triangle_vertices[0,:],
                       triangle_vertices[10,:],
                       triangle_vertices[4,:],
                       triangle_vertices[9,:]))

print(scrambled)

if plot:

    plt.figure(figsize=(8, 8))
    plt.scatter(scrambled[:, 0], scrambled[:, 1], color='blue')
    plt.title('Triangle Vertices')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(0, 8)
    plt.ylim(0, 6)
    plt.show()