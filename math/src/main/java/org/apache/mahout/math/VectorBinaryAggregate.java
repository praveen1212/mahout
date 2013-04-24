package org.apache.mahout.math;

import com.google.common.base.Preconditions;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.util.Iterator;

public abstract class VectorBinaryAggregate {
  private static final VectorBinaryAggregate operations[] = new VectorBinaryAggregate[] {
      // case 1
      new AggregateIterateOneLookupOther(true),
      new AggregateIterateOneLookupOther(false),
      // case 2
      new AggregateIterateIntersection(),
      // case 3
      new AggregateIterateUnionSequential(),
      new AggregateIterateUnionRandom(),
      // case 4
      new AggregateIterateAllSequential(),
      new AggregateIterateAllLookup(true),
      new AggregateIterateAllLookup(false),
      new AggregateAllRandom(),
  };

  public abstract boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc);

  public abstract double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc);

  public abstract double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc);

  public static double aggregateBest(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
    VectorBinaryAggregate bestOperation = null;
    double bestCost = Double.POSITIVE_INFINITY;
    for (int i = 0; i < operations.length; ++i) {
      if (operations[i].isValid(x, y, fa, fc)) {
        double cost = operations[i].estimateCost(x, y, fa, fc);
        if (cost < bestCost) {
          bestCost = cost;
          bestOperation = operations[i];
        }
      }
    }
    Preconditions.checkNotNull(bestOperation, "No valid operation for vector assign");
    return bestOperation.aggregate(x, y, fa, fc);
  }

  public static class AggregateIterateOneLookupOther extends VectorBinaryAggregate {
    private final boolean swap;

    public AggregateIterateOneLookupOther(boolean swap) {
      this.swap = swap;
    }

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && (fa.isAssociativeAndCommutative() || x.isSequentialAccess())
          && fc.isLikeLeftMult();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.getNumNondefaultElements() * x.getIterateNonzeroAdvanceTime() * y.getRandomAccessLookupTime();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Vector.Element xe;
      double result = 0;
      boolean validResult = false;
      while (xi.hasNext()) {
        xe = xi.next();
        double yv = y.getQuick(xe.index());
        double thisResult;
        if (!swap) {
          thisResult = fc.apply(xe.get(), yv);
        } else {
          thisResult = fc.apply(yv, xe.get());
        }
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
      }
      return result;
    }
  }

  public static class AggregateIterateIntersection extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && fc.isLikeMult() && x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.getNumNondefaultElements() * x.getIterateNonzeroAdvanceTime()
          + y.getNumNondefaultElements() * y.getIterateNonzeroAdvanceTime();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
      double result = 0;
      boolean validResult = false;
      while (xi.hasNext() && yi.hasNext()) {
        if (advanceThis) {
          xe = xi.next();
        }
        if (advanceThat) {
          ye = yi.next();
        }
        if (xe.index() == ye.index()) {
          double thisResult = fc.apply(xe.get(), ye.get());
          if (validResult) {
            result = fa.apply(result, thisResult);
          } else {
            result = thisResult;
            validResult = true;
          }
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
      return result;
    }
  }

  public static class AggregateIterateUnionSequential extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && !fc.isDensifying() && x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.getNumNondefaultElements() * x.getIterateNonzeroAdvanceTime()
          + y.getNumNondefaultElements() * y.getIterateNonzeroAdvanceTime();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
      double result = 0;
      boolean validResult = false;

      while (advanceThis || advanceThat) {
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
        double thisResult;
        if (xe != null && ye != null) { // both vectors have nonzero elements
          if (xe.index() == ye.index()) {
            thisResult = fc.apply(xe.get(), ye.get());
            advanceThis = true;
            advanceThat = true;
          } else {
            if (xe.index() < ye.index()) { // f(x, 0)
              thisResult = fc.apply(xe.get(), 0);
              advanceThis = true;
              advanceThat = false;
            } else {
              thisResult = fc.apply(0, ye.get());
              advanceThis = false;
              advanceThat = true;
            }
          }
        } else if (xe != null) { // just the first one still has nonzeros
          thisResult = fc.apply(xe.get(), 0);
          advanceThis = true;
          advanceThat = false;
        } else if (ye != null) { // just the second one has nonzeros
          thisResult = fc.apply(0, ye.get());
          advanceThis = false;
          advanceThat = true;
        } else { // we're done, both are empty
          break;
        }
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
      }
      return result;
    }
  }

  public static class AggregateIterateUnionRandom extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && fa.isAssociativeAndCommutative() && !fc.isDensifying();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.getNumNondefaultElements() * x.getIterateNonzeroAdvanceTime() * y.getRandomAccessLookupTime()
          + y.getNumNondefaultElements() * y.getIterateNonzeroAdvanceTime() * x.getRandomAccessLookupTime();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      OpenIntHashSet visited = new OpenIntHashSet();
      Iterator<Vector.Element> xi = x.iterateNonZero();
      Vector.Element xe;
      double result = 0;
      boolean validResult = false;
      while (xi.hasNext()) {
        xe = xi.next();
        double thisResult = fc.apply(xe.get(), y.getQuick(xe.index()));
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else  {
          result = thisResult;
          validResult = true;
        }
        visited.add(xe.index());
      }
      Iterator<Vector.Element> yi = y.iterateNonZero();
      Vector.Element ye;
      while (yi.hasNext()) {
        ye = yi.next();
        if (!visited.contains(ye.index())) {
          double thisResult = fc.apply(x.getQuick(ye.index()), ye.get());
          if (validResult) {
            result = fa.apply(result, thisResult);
          } else  {
            result = thisResult;
            validResult = true;
          }
        }
      }
      return result;
    }
  }

  public static class AggregateIterateAllSequential extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.size() + y.size();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.iterator();
      Iterator<Vector.Element> yi = y.iterator();
      Vector.Element xe;
      double result = 0;
      boolean validResult = false;
      while (xi.hasNext() && yi.hasNext()) {
        xe = xi.next();
        double thisResult = fc.apply(xe.get(), yi.next().get());
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
      }
      return result;
    }
  }

  public static class AggregateIterateAllLookup extends VectorBinaryAggregate {
    private final boolean swap;

    public AggregateIterateAllLookup(boolean swap) {
      this.swap = swap;
    }

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isAssociativeAndCommutative() || x.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.size() * y.getRandomAccessLookupTime();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.iterator();
      Vector.Element xe;
      double result = 0;
      boolean validResult = false;
      while (xi.hasNext()) {
        xe = xi.next();
        double thisResult;
        if (!swap) {
          thisResult = fc.apply(xe.get(), y.getQuick(xe.index()));
        } else {
          thisResult = fc.apply(y.getQuick(xe.index()), xe.get());
        }
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
      }
      return result;
    }
  }

  public static class AggregateAllRandom extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return true;
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.size() * x.getRandomAccessLookupTime() * y.getRandomAccessLookupTime();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      double result = fc.apply(x.getQuick(0), y.getQuick(0));
      for (int i = 1; i < x.size(); ++i) {
        result = fa.apply(result, fc.apply(x.getQuick(i), y.getQuick(i)));
      }
      return result;
    }
  }
}
