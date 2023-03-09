import math
from sklearn.datasets._samples_generator import make_blobs
import numpy as np
import scipy.stats as st




X_surface, truth_surface = make_blobs(n_samples=80000, centers=[[150, 200], [150, 800], [400, 500], [800, 250], [800, 700]],
                                      cluster_std=[80, 80, 70, 80, 100], random_state=42)
x_surface = X_surface[:, 0]
y_surface = X_surface[:, 1]
xx_surface, yy_surface = np.mgrid[0:951.5:23j,0:973.7:14j]
positions_surface = np.vstack([xx_surface.ravel(), yy_surface.ravel()])
values_surface = np.vstack([x_surface, y_surface])
kernel_surface = st.gaussian_kde(values_surface)
f_surface = np.reshape(kernel_surface(positions_surface).T, xx_surface.shape)
zz_surface = f_surface * 30000000

#------------------------------------------------------------------------------#
X_sensor2, y_true_sensor2 = make_blobs(n_samples=1800, centers=[[250, 750], [605, 225]],
                                     cluster_std=[150,150], random_state=42)
x_sensors2 = X_sensor2[:, 0]
y_sensors2 = X_sensor2[:, 1]
xx_sensors2, yy_sensors2 = np.mgrid[0:1000:50j,0:1000:50j]
positions_sensors2 = np.vstack([xx_sensors2.ravel(), yy_sensors2.ravel()])
values_sensor2 = np.vstack([x_sensors2, y_sensors2])
kernel_sensor2 = st.gaussian_kde(values_sensor2)
f_sensor2 = np.reshape(kernel_sensor2(positions_sensors2).T, xx_sensors2.shape)
zz_sensor2 = f_sensor2 * 2332800 # 4000 sensors
# zz_sensor2 = f_sensor2 * 1839900 # 3000 sensors
# zz_sensor2 = f_sensor2 * 1377500 # 2000 sensors

vector_sensor2 = np.vectorize(np.int32)
sensor_number2 = vector_sensor2(zz_sensor2)  #

X_sensor3, y_true_sensor3 = make_blobs(n_samples=2200, centers=[[150, 225],[685, 750], [735, 450]],
                                     cluster_std=[180,180, 180], random_state=42)
x_sensors3 = X_sensor3[:, 0]
y_sensors3 = X_sensor3[:, 1]
xx_sensors3, yy_sensors3 = np.mgrid[0:1000:50j,0:1000:50j]
positions_sensors3 = np.vstack([xx_sensors3.ravel(), yy_sensors3.ravel()])
values_sensor3 = np.vstack([x_sensors3, y_sensors3])
kernel_sensor3 = st.gaussian_kde(values_sensor3)
f_sensor3 = np.reshape(kernel_sensor3(positions_sensors3).T, xx_sensors3.shape)
zz_sensor3 = f_sensor3 * 3623500 # 6000 sensors
# zz_sensor3 = f_sensor3 * 3109000 # 5000 sensors
# zz_sensor3 = f_sensor3 * 2100850 # 3000 sensors
vector_sensor3 = np.vectorize(np.int32)
sensor_number3 = vector_sensor3(zz_sensor3)  #


hexagon_coordinates = dict()
hexagon_heights = dict()
state_index = 0
x_value = 43.25 # i have divided the x axis into 23 951.5 / 43.25
y_value = 74.9 # 973.7 / 74.9
fixed_height = 150
init_x = 86.5
init_y = 74.9
environment_rows = 6
environment_columns = 7
height_init_y = 1
for y in range(environment_rows):
    y_co = init_y
    x_co = init_x
    y_height_init = height_init_y
    x_height_init = 2
    for x in range(environment_columns):
        hexagon_coordinates[state_index] = [round(x_co, 2), round(y_co,2)]
        hexagon_heights[state_index] = zz_surface[x_height_init][y_height_init] + fixed_height
        if (x % 2) != 0:
            y_co -= y_value
            x_co += 3*x_value
            y_height_init -= 1
            x_height_init += 3
        else:
            y_co += y_value
            x_co += 3*x_value
            y_height_init += 1
            x_height_init += 3
        if x == environment_columns-1:
            init_y += 2 * y_value
            height_init_y += 2
        state_index += 1

long_diameter = 173 # hexagon long diaameter
# NewHexagonID = list(hexagon_coordinates.keys())[
#             list(hexagon_coordinates.values()).index([216.25,299.6])]
# print(NewHexagonID)
def vertices_coordinate(x, y):
    verices_list = []
    alha = long_diameter / 4
    beta = math.sqrt(3) * alha
    a = [x+2*alha, y]
    b = [x+alha, y + beta]
    c = [x- alha, y + beta]
    d = [x-2*alha, y]
    e = [x-alha, y- beta ]
    f = [x+ alha, y- beta]
    verices_list.append(a)
    verices_list.append(b)
    verices_list.append(c)
    verices_list.append(d)
    verices_list.append(e)
    verices_list.append(f)
    return verices_list

RIGHT = "RIGHT"
LEFT = "LEFT"

def inside_convex_polygon(x, y, point):
    previous_side = None
    vertices = vertices_coordinate(x, y)
    n_vertices = len(vertices)
    for n in range(n_vertices):
        a, b = vertices[n], vertices[(n+1)%n_vertices]
        affine_segment = v_sub(b, a)
        affine_point = v_sub(point, a)
        current_side = get_side(affine_segment, affine_point)
        if current_side is None:
            return False #outside or over an edge
        elif previous_side is None: #first segment
            previous_side = current_side
        elif previous_side != current_side:
            return False
    return True

def get_side(a, b):
    x = cosine_sign(a, b)
    if x < 0:
        return LEFT
    elif x > 0:
        return RIGHT
    else:
        return None

def v_sub(a, b):
    return (a[0]-b[0], a[1]-b[1])

def cosine_sign(a, b):
    return a[0]*b[1]-a[1]*b[0]

# sensor_cells_info = dict()
# cell_size = 20
# for i in range(50):
#     for j in range(50):
#         sensor_cells_info[i*cell_size,j*cell_size] = [sensor_number2[j][i], sensor_number3[j][i]]
# print('sensor cell info', sensor_cells_info)
#
# Sensors_in_hexagon = dict() # use as a number of sensors
# for index_state in range(1, 43):
#     sensor_sum = [0,0,0]
#
#     for i in sensor_cells_info:
#         StateValue = hexagon_coordinates[index_state]
#         if inside_convex_polygon(StateValue[1], StateValue[0], [i[0], i[1]]) is True:
#             sensor_sum[0] += sensor_cells_info[i][0]
#             sensor_sum[1] += sensor_cells_info[i][1]
#             sensor_sum[2] += sensor_cells_info[i][2]
#     Sensors_in_hexagon[index_state] = sensor_sum
# print("Sensor in hexagon", Sensors_in_hexagon)
# sensor1= 0
# sensor2= 0
cell_size = 20
Sensors_in_hexagon = dict() # use as a number of sensors
for index_state in range(42):
    sensor_sum = [0,0]
    for i in range(50):
        for j in range(50):
            StateValue = hexagon_coordinates[index_state]
            if inside_convex_polygon(StateValue[0], StateValue[1], [i * cell_size, j * cell_size]) is True:
                sensor_sum[0] += sensor_number2[i][j]
                sensor_sum[1] += sensor_number3[i][j]
            # if index_state == 1:
            #     sensor1 += sensor_number2[i][j]
            #     sensor2 += sensor_number3[i][j]
    Sensors_in_hexagon[index_state] = sensor_sum
# print("Sensor in hexagon", Sensors_in_hexagon)
# print('hexagon coordinates', hexagon_coordinates)
# print('hexagon heights', hexagon_heights)
# print(sensor1, sensor2)
#
#
# sum_val1 = 0
# sum_val2 = 0
# for value in Sensors_in_hexagon.values():
#     sum_val1 += value[0]
#     sum_val2 += value[1]
# print('sensor in hexagon', sum_val1, sum_val2)
