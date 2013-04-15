package org.apache.mahout.math.function;

/**
 * Empty interface that should be implemented by functions f(x, y) where:
 * f(0, y) = 0 for any y.
 *
 * This specialized interface enables faster vector operations.
 * If a function has this property, it should implement this interface.
 *
 * This is true for functions like MULT: 0 * y = 0.
 */
public interface LikeLeftMult extends DoubleDoubleFunction {
}
