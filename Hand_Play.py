import time
import gym
from pynput import keyboard
import threading

a = 1
def keyboard_release(key):
    global a
    if key == keyboard.Key.left:
        a = 0
    if key == keyboard.Key.down:
        a = 1
    if key == keyboard.Key.right:
        a = 2
    time.sleep(0.5)
    a = 1

def keyboardListener():
    with keyboard.Listener(
            on_release=keyboard_release) as listener:
        listener.join()

env = gym.make("Acrobot-v1")
env = env.unwrapped
tStart = time.time()
k = env.reset()
threading._start_new_thread(keyboardListener, ())
while True:
    s_, r, done, info = env.step(a)
    if done:
        tEnd = time.time()  # time counter End
        print("Time:", ("%.2f" % (tEnd - tStart)))
        time.sleep(2)
        break
    env.render()
env.close()