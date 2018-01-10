



# takes the input vectors x and w where x is the vector containing inputs to a neuron and w is a vector containing weights
# to each input (signalling the strength of each connection). Finally, b is a constant term known as a bias.
def integration(x,w,b):
    weighted_sum = sum(x[k] * w[k] for k in xrange(0,len(x)))
    return weighted_sum + b


# correctly computes this magnitude given an input vector as a list
def magnitude(x):
    return sum(k**2 for k in x)**0.5


# Given a function gradient that has computed the gradient for a given function (and the ability to do vector addition),
# pseudocode for gradient descent would look like this:
def gradient_descent(point,step_size,threshold):
    value = f(point)
    new_point = point - step_size * gradient(point)
    new_value = f(new_point)
    if abs(new_value - value) < threshold:
        return value
    return gradient_descent(new_point,step_size,threshold)