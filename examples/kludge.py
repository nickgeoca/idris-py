import tensorflow as tf


def while_loop(cond, body, loop_vars, parallel_iterations=10, back_prop=True, swap_memory=False, name=None):
    def cond2(*ls): return cond(ls)
    def body2(*ls): return body(ls)
    return tf.while_loop(cond2, body2, loop_vars, parallel_iterations, back_prop, swap_memory, name)
