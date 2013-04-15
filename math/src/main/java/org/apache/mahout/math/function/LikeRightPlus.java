package org.apache.mahout.math.function;

/**
 * Empty interface that should be implemented by functions f(x, y) where:
 * f(x, 0) = x for any x.
 *
 * This specialized interface enables faster vector operations.
 * If a function has this property, it should implement this interface.
 *
 * This is true for functions like PLUS: x + 0 = x.
 */
public interface LikeRightPlus extends DoubleDoubleFunction {
}
