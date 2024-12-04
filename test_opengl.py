from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import glfw
import assimp
import time

path = '/Users/ed/Data/3D/Armadillo/Armadillo.ply'
data = assimp.load(path)


# glfw.create_window("dd")
glfw.init()
w = glfw.create_window(800, 600, "hello, glfw", None, None)
if w is None:
    print("! creat window fail.")
    exit(-1)

glfw.make_context_current(w)
glfw.swap_interval(1)

while True:
    time.sleep(1)

glfw.destroy_window(w)
glfw.terminate()