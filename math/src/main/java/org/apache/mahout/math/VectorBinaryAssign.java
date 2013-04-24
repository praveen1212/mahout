package org.apache.mahout.math;

import com.google.common.base.Preconditions;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.util.Iterator;

public abstract class VectorBinaryAssign {
  private static final VectorBinaryAssign operations[] = new VectorBinaryAssign[] {
      // case 1
      new AssignIterateOneLookupOther(true),  // 0
      new AssignIterateOneLookupOther(false),  // 1
      // case 2
      new AssignIterateIntersection(),  // 2
      // case 3
      new AssignIterateUnionSequential(),  // 3
      new AssignIterateUnionRandom(),  // 4
      // case 4
      new AssignIterateAllSequential(),  // 5
      new AssignIterateAllLookup(true),  // 6
      new AssignIterateAllLookup(false),  // 7
      new AssignAllRandom(),  // 8
  };

  public abstract boolean isValid(Vector x, Vector y, DoubleDoubleFunction f);

  public abstract double estimateCost(Vector x, Vector y, DoubleDoubleFunction f);

  public abstract Vector assign(Vector x, Vector y, DoubleDoubleFunction f);

  public static Vector assignBest(Vector x, Vector y, DoubleDoubleFunction f) {
    int bestOperationIndex = -1;
    double bestCost = Double.POSITIVE_INFINITY;
    for (int i = 0; i < operations.length; ++i) {
      if (operations[i].isValid(x, y, f)) {
        double cost = operations[i].estimateCost(x, y, f);
        if (cost < bestCost) {
          bestCost = cost;
          bestOperationIndex = i;
        }
      }
    }
    Preconditions.checkArgument(bestOperationIndex >= 0, "No valid operation for vector assign");
    return operations[bestOperationIndex].assign(x, y, f);
  }

  public static class AssignIterateOneLookupOther extends VectorBinaryAssign {
    /**
     * False if iterating through x and looking up in y.
     * True if iterating through y and looking up in x.
     */
    private final boolean swap;

    public AssignIterateOneLookupOther(boolean swap) {
      this.swap = swap;
    }

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      /**
       * swap == false: iterate through x iff f(0, y) = 0
       * swap == true: iterate through y iff f(x, 0) = x
       */
      if (!swap) {
        return f.isLikeLeftMult();
      } else {
        // If x can't insert new elements in constant time, we need to use the merge updates trick
        // and the merged updates OrderedIntDoubleMappings needs to be ordered, therefore y has to support
        // sequential access.
        boolean thisSSAVthatSequential = x.isAddConstantTime() || y.isSequentialAccess();
        return f.isLikeRightPlus() && thisSSAVthatSequential;
      }
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return !swap ? x.getNumNondefaultElements() * x.getIterateNonzeroAdvanceTime() * y.getRandomAccessLookupTime()
          : y.getNumNondefaultElements() * y.getIterateNonzeroAdvanceTime() * x.getRandomAccessLookupTime();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      return !swap ? assignInner(x, y, f) : assignInner(y, x, f);
    }

    public Vector assignInner(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Vector.Element xe;
      Vector.Element ye;
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
      while (xi.hasNext()) {
        xe = xi.next();
        ye = y.getElement(xe.index());
        if (!swap) {
          xe.set(f.apply(xe.get(), ye.get()));
        } else {
          if (ye.get() != 0.0 || y.isAddConstantTime()) {
            ye.set(f.apply(ye.get(), xe.get()));
          } else {
            updates.set(xe.index(), f.apply(ye.get(), xe.get()));
          }
        }
      }
      if (swap && !y.isAddConstantTime()) {
        y.mergeUpdates(updates);
      }
      return swap ? y : x;
    }
  }

  public static class AssignIterateIntersection extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return f.isLikeLeftMult() && f.isLikeRightPlus();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.max(x.getNumNondefaultElements() * x.getIterateNonzeroAdvanceTime(),
          y.getNumNondefaultElements() * y.getIterateNonzeroAdvanceTime());
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
      while (xi.hasNext() && yi.hasNext()) {
        if (advanceThis) {
          xe = xi.next();
        }
        if (advanceThat) {
          ye = yi.next();
        }
        if (xe.index() == ye.index()) {
          xe.set(f.apply(xe.get(), ye.get()));
          advanceThis = true;
          advanceThat = true;
        } else {
          if (xe.index() < ye.index()) { // f(x, 0) = 0
            advanceThis = true;
            advanceThat = false;
          } else { // f(0, y) = 0
            advanceThis = false;
            advanceThat = true;
          }
        }
      }
      return x;
    }
  }

  public static class AssignIterateUnionSequential extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return !f.isDensifying() && x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.max(x.getNumNondefaultElements() * x.getIterateNonzeroAdvanceTime(),
          y.getNumNondefaultElements() * y.getIterateNonzeroAdvanceTime());
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
      OrderedIntDoubleMapping thisUpdates = new OrderedIntDoubleMapping();

      while (true) {
        if (advanceThis) {
          if (xi.hasNext()) {
            xe = xi.next();
          } else {
            xe = null;
          }
        }
        if (advanceThat) {
          if (yi.hasNext()) {
            ye = yi.next();
          } else {
            ye = null;
          }
        }
        if (xe != null && ye != null) { // both vectors have nonzero elements
          if (xe.index() == ye.index()) {
            xe.set(f.apply(xe.get(), ye.get()));
            advanceThis = true;
            advanceThat = true;
          } else {
            if (xe.index() < ye.index()) { // f(x, 0)
              xe.set(f.apply(xe.get(), 0));
              advanceThis = true;
              advanceThat = false;
            } else {
              double result = f.apply(0, ye.get());
              if (result != 0) { // f(0, y) != 0
                if (x.isAddConstantTime()) {
                  x.setQuick(ye.index(), result);
                } else {
                  thisUpdates.set(ye.index(), result);
                }
              }
              advanceThis = false;
              advanceThat = true;
            }
          }
        } else if (xe != null) { // just the first one still has nonzeros
          xe.set(f.apply(xe.get(), 0));
          advanceThis = true;
          advanceThat = false;
        } else if (ye != null) { // just the second one has nonzeros
          double result = f.apply(0, ye.get());
          if (result != 0) {
            if (x.isAddConstantTime()) {
              x.setQuick(ye.index(), result);
            } else {
              thisUpdates.set(ye.index(), result);
            }
          }
          advanceThis = false;
          advanceThat = true;
        } else { // we're done, both are empty
          break;
        }
      }
      if (!x.isAddConstantTime()) {
        x.mergeUpdates(thisUpdates);
      }
      return x;
    }
  }

  public static class AssignIterateUnionRandom extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return !f.isDensifying() && (x.isAddConstantTime() || y.isSequentialAccess());
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.max(x.getNumNondefaultElements() * x.getIterateNonzeroAdvanceTime() * y.getRandomAccessLookupTime(),
          y.getNumNondefaultElements() * y.getIterateNonzeroAdvanceTime() * x.getRandomAccessLookupTime());
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      OpenIntHashSet visited = new OpenIntHashSet();
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Vector.Element xe;
      while (xi.hasNext()) {
        xe = xi.next();
        double result = f.apply(xe.get(), y.getQuick(xe.index()));
        xe.set(result);
        visited.add(xe.index());
      }
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element ye;
      while (yi.hasNext()) {
        ye = yi.next();
        if (!visited.contains(ye.index())) {
          double result = f.apply(x.getQuick(ye.index()), ye.get());
          if (x.isAddConstantTime()) {
            x.setQuick(ye.index(), result);
          } else {
            updates.set(ye.index(), result);
          }
        }
      }
      if (!x.isAddConstantTime()) {
        x.mergeUpdates(updates);
      }
      return x;
    }
  }

  public static class AssignIterateAllSequential extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.size();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.iterator();
      Iterator<Vector.Element> yi = y.iterator();
      Vector.Element xe;
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
      while (xi.hasNext() && yi.hasNext()) {
        xe = xi.next();
        double result = f.apply(xe.get(), yi.next().get());
        if (x.isAddConstantTime()) {
          x.setQuick(xe.index(), result);
        } else {
          updates.set(xe.index(), result);
        }
      }
      if (!x.isAddConstantTime()) {
        x.mergeUpdates(updates);
      }
      return x;
    }
  }

  public static class AssignIterateAllLookup extends VectorBinaryAssign {
    private final boolean swap;

    public AssignIterateAllLookup(boolean swap) {
      this.swap = swap;
    }

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return true;
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return !swap ? x.size() * y.getRandomAccessLookupTime() : y.size() * x.getRandomAccessLookupTime();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      if (!swap) {
        return assignInner(x, y, f);
      } else {
        return assignInner(y, x, f);
      }
    }

    public Vector assignInner(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.iterator();
      Vector.Element xe;
      Vector.Element ye;
      double result;
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
      while (xi.hasNext()) {
        xe = xi.next();
        ye = y.getElement(xe.index());
        if (!swap) {
          result = f.apply(xe.get(), ye.get());
          if (result != xe.get()) {
            if (xe.get() != 0 || x.isAddConstantTime()) {
              xe.set(result);
            } else  {
              Preconditions.checkArgument(xe.get() == 0);
              updates.set(xe.index(), result);
            }
          }
        } else {
          result = f.apply(ye.get(), xe.get());
          if (result != ye.get()) {
            if (ye.get() != 0.0 || y.isAddConstantTime()) {
              ye.set(result);
            } else {
              updates.set(xe.index(), result);
            }
          }
        }
      }
      if (!swap) {
        if (!x.isAddConstantTime()) {
          x.mergeUpdates(updates);
        }
      } else {
        if (!y.isAddConstantTime()) {
          y.mergeUpdates(updates);
        }
      }
      return swap ? y : x;
    }
  }

  public static class AssignAllRandom extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return true;
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.size() * x.getRandomAccessLookupTime() * y.getRandomAccessLookupTime();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
      for (int i = 0; i < x.size(); ++i) {
        double result = f.apply(x.getQuick(i), y.getQuick(i));
        if (x.isAddConstantTime()) {
          x.setQuick(i, result);
        } else {
          updates.set(i, result);
        }
      }
      if (!x.isAddConstantTime()) {
        x.mergeUpdates(updates);
      }
      return x;
    }
  }
}
